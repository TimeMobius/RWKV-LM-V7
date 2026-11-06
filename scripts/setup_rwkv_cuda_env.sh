#!/usr/bin/env bash

# Source this file before launching train.py.  Conda's CUDA packages keep the
# compiler in $CONDA_PREFIX/bin but split headers and libraries across the
# targets tree and pip's nvidia/* packages; PyTorch's extension helper does not
# discover all of those directories automatically after a host reboot.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source scripts/setup_rwkv_cuda_env.sh" >&2
  exit 2
fi

_rwkv_setup_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_rwkv_repo_dir="$(cd "${_rwkv_setup_dir}/.." && pwd)"

# Keep CUDA extensions and the training process on the same PyTorch install.
# Callers can select another interpreter explicitly with PYTHON_BIN.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/mnt/data/anaconda3/envs/rwkv-lm-v7/bin/python" ]]; then
    PYTHON_BIN="/mnt/data/anaconda3/envs/rwkv-lm-v7/bin/python"
  else
    PYTHON_BIN="$(command -v python || true)"
  fi
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find a Python interpreter; set PYTHON_BIN explicitly." >&2
  return 2
fi
if [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not an executable Python interpreter: ${PYTHON_BIN:-<empty>}" >&2
  return 2
fi
export PYTHON_BIN

if [[ -z "${TORCH_EXTENSIONS_DIR:-}" ]]; then
  TORCH_EXTENSIONS_DIR="${_rwkv_repo_dir}/.torch_extensions"
fi
export TORCH_EXTENSIONS_DIR

# When the interpreter is selected by absolute path, make its companion CUDA
# compiler discoverable even when the caller did not run `conda activate`.
_rwkv_python_bin_dir="$(dirname "${PYTHON_BIN}")"
if [[ -d "${_rwkv_python_bin_dir}" ]]; then
  case ":${PATH}:" in
    *":${_rwkv_python_bin_dir}:"*) ;;
    *) export PATH="${_rwkv_python_bin_dir}:${PATH}" ;;
  esac
fi

# DeepSpeed/PyTorch 2.7 can eagerly create a large Inductor worker pool during
# import. These training paths use explicit CUDA extensions, so keep startup
# bounded while allowing callers to override the defaults.
export TORCH_WARM_POOL="${TORCH_WARM_POOL:-0}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

_rwkv_cuda_python="${PYTHON_BIN}"
_rwkv_cuda_nvcc="${RWKV_CUDACXX:-$(command -v nvcc || true)}"
if [[ -z "${_rwkv_cuda_nvcc}" ]]; then
  echo "Could not find nvcc on PATH; activate the CUDA-enabled training environment first." >&2
  return 2
fi

export CUDA_HOME="$(cd "$(dirname "${_rwkv_cuda_nvcc}")/.." && pwd)"
export CUDACXX="${_rwkv_cuda_nvcc}"

_rwkv_cuda_site="$(${_rwkv_cuda_python} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
_rwkv_torch_lib="$(${_rwkv_cuda_python} -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")' 2>/dev/null || true)"
_rwkv_cuda_include="${CUDA_HOME}/include:${CUDA_HOME}/targets/x86_64-linux/include"
_rwkv_cuda_lib="${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib"
if [[ -n "${_rwkv_cuda_site}" && -d "${_rwkv_cuda_site}/nvidia" ]]; then
  while IFS= read -r _rwkv_dir; do
    _rwkv_cuda_include="${_rwkv_dir}:${_rwkv_cuda_include}"
  done < <(find "${_rwkv_cuda_site}/nvidia" -type d -name include -print 2>/dev/null)
  while IFS= read -r _rwkv_dir; do
    _rwkv_cuda_lib="${_rwkv_dir}:${_rwkv_cuda_lib}"
  done < <(find "${_rwkv_cuda_site}/nvidia" -type d -name lib -print 2>/dev/null)
fi
if [[ -n "${_rwkv_torch_lib}" && -d "${_rwkv_torch_lib}" ]]; then
  _rwkv_cuda_lib="${_rwkv_torch_lib}:${_rwkv_cuda_lib}"
fi

# pip CUDA wheels intentionally ship versioned shared objects only.  The
# DeepSpeed JIT linker asks for conventional names such as -lcurand and
# -lcublas, so provide recoverable compatibility links in /tmp rather than
# modifying the conda environment.
_rwkv_cuda_link_dir="${RWKV_CUDA_LINK_DIR:-/tmp/rwkv_cuda_links}"
mkdir -p "${_rwkv_cuda_link_dir}"
if [[ -n "${_rwkv_cuda_site}" && -d "${_rwkv_cuda_site}/nvidia" ]]; then
  while IFS= read -r _rwkv_lib_file; do
    _rwkv_lib_base="$(basename "${_rwkv_lib_file}")"
    _rwkv_lib_name="${_rwkv_lib_base%%.so.*}.so"
    [[ "${_rwkv_lib_base}" == *.so.* ]] || continue
    [[ -e "${_rwkv_cuda_link_dir}/${_rwkv_lib_name}" ]] || ln -s "${_rwkv_lib_file}" "${_rwkv_cuda_link_dir}/${_rwkv_lib_name}"
  done < <(find "${_rwkv_cuda_site}/nvidia" -type f -name 'lib*.so.*' -print 2>/dev/null)
  _rwkv_cuda_lib="${_rwkv_cuda_link_dir}:${_rwkv_cuda_lib}"
fi

export CPATH="${_rwkv_cuda_include}${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${_rwkv_cuda_include}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="${_rwkv_cuda_lib}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${_rwkv_cuda_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# FileBaton locks are held by an open descriptor while an extension is being
# built. A killed process can leave only the path behind, making later ranks
# spin forever. Remove only old, unheld locks under this explicit directory.
_rwkv_clean_locks="${RWKV_CLEAN_STALE_EXTENSION_LOCKS:-1}"
_rwkv_lock_age_limit="${RWKV_EXTENSION_LOCK_STALE_SECONDS:-1800}"
if [[ "${_rwkv_clean_locks}" == "1" && -d "${TORCH_EXTENSIONS_DIR}" ]]; then
  if [[ "${_rwkv_lock_age_limit}" =~ ^[0-9]+$ ]]; then
    _rwkv_now="$(date +%s)"
    while IFS= read -r -d '' _rwkv_lock; do
      _rwkv_mtime="$(stat -c %Y "${_rwkv_lock}" 2>/dev/null || true)"
      [[ "${_rwkv_mtime}" =~ ^[0-9]+$ ]] || continue
      if (( _rwkv_now - _rwkv_mtime < _rwkv_lock_age_limit )); then
        continue
      fi
      if fuser -s "${_rwkv_lock}" 2>/dev/null; then
        continue
      fi
      rm -f -- "${_rwkv_lock}"
      echo "Removed stale CUDA extension lock: ${_rwkv_lock}" >&2
    done < <(find "${TORCH_EXTENSIONS_DIR}" -type f -name lock -print0 2>/dev/null)
  else
    echo "Ignoring invalid RWKV_EXTENSION_LOCK_STALE_SECONDS=${_rwkv_lock_age_limit}; expected a non-negative integer." >&2
  fi
fi

unset _rwkv_setup_dir _rwkv_repo_dir _rwkv_python_bin_dir _rwkv_cuda_python _rwkv_cuda_nvcc _rwkv_cuda_site _rwkv_torch_lib _rwkv_cuda_include _rwkv_cuda_lib _rwkv_cuda_link_dir _rwkv_lib_file _rwkv_lib_base _rwkv_lib_name _rwkv_dir _rwkv_clean_locks _rwkv_lock_age_limit _rwkv_now _rwkv_lock _rwkv_mtime

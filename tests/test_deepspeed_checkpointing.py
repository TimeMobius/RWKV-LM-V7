import os
import sys
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train
from src import dataset as dataset_mod
from src import trainer as trainer_mod

_REGRESSION_SCRIPT_PATH = REPO_ROOT / "scripts" / "regression_resume_deepspeed_checkpoint.py"
_REGRESSION_SCRIPT_SPEC = importlib.util.spec_from_file_location("regression_resume_deepspeed_checkpoint", _REGRESSION_SCRIPT_PATH)
assert _REGRESSION_SCRIPT_SPEC is not None and _REGRESSION_SCRIPT_SPEC.loader is not None
resume_smoke = importlib.util.module_from_spec(_REGRESSION_SCRIPT_SPEC)
_REGRESSION_SCRIPT_SPEC.loader.exec_module(resume_smoke)


def _make_ds_checkpoint_dir(base: Path, name: str, ts: int) -> Path:
    path = base / name
    path.mkdir()
    (path / "latest").write_text("checkpoint", encoding="utf-8")
    (path / "zero_pp_rank_0_mp_rank_00_model_states.pt").write_text("", encoding="utf-8")
    os.utime(path / "latest", (ts, ts))
    os.utime(path / "zero_pp_rank_0_mp_rank_00_model_states.pt", (ts, ts))
    os.utime(path, (ts, ts))
    return path


def _make_file_checkpoint(base: Path, name: str, ts: int) -> Path:
    path = base / name
    path.write_bytes(b"checkpoint")
    os.utime(path, (ts, ts))
    return path


def test_parse_epoch_checkpoint_name_only_accepts_epoch_style_names():
    assert train.parse_epoch_checkpoint_name("rwkv-init.pth") == -1
    assert train.parse_epoch_checkpoint_name("rwkv-7.pth") == 7
    assert train.parse_epoch_checkpoint_name("rwkv-step-7.pth") is None
    assert train.parse_epoch_checkpoint_name("not-a-checkpoint.pth") is None


def test_parse_step_checkpoint_name_only_accepts_step_style_names():
    assert train.parse_step_checkpoint_name("rwkv-step-7.pth") == 7
    assert train.parse_step_checkpoint_name("rwkv-7.pth") is None
    assert train.parse_step_checkpoint_name("rwkv-init.pth") is None
    assert train.parse_step_checkpoint_name("not-a-checkpoint.pth") is None


def test_find_stage2_checkpoint_uses_init_when_no_epoch_checkpoint_exists(tmp_path: Path):
    init_path = tmp_path / "rwkv-init.pth"
    init_path.write_bytes(b"checkpoint")

    path, max_slot, previous_slot = train.find_stage2_checkpoint(str(tmp_path))

    assert path == str(init_path)
    assert max_slot == -1
    assert previous_slot is None


def test_find_stage2_checkpoint_reports_empty_output_without_init(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no rwkv-init.pth fallback"):
        train.find_stage2_checkpoint(str(tmp_path))


def test_find_stage2_checkpoint_has_no_invalid_previous_slot_for_single_checkpoint(tmp_path: Path):
    checkpoint = _make_file_checkpoint(tmp_path, "rwkv-7.pth", 100)

    path, max_slot, previous_slot = train.find_stage2_checkpoint(str(tmp_path))

    assert path == str(checkpoint)
    assert max_slot == 7
    assert previous_slot is None


def test_detects_deepspeed_checkpoint_directories(tmp_path: Path):
    ds_dir = _make_ds_checkpoint_dir(tmp_path, "rwkv-1.pth", 100)
    assert train.is_deepspeed_checkpoint_dir(str(ds_dir))

    missing_marker_dir = tmp_path / "rwkv-2.pth"
    missing_marker_dir.mkdir()
    assert not train.is_deepspeed_checkpoint_dir(str(missing_marker_dir))

    wrong_suffix_dir = tmp_path / "rwkv-3"
    wrong_suffix_dir.mkdir()
    (wrong_suffix_dir / "latest").write_text("checkpoint", encoding="utf-8")
    assert not train.is_deepspeed_checkpoint_dir(str(wrong_suffix_dir))


@pytest.mark.parametrize(
    "strategy",
    [
        "deepspeed_stage_1",
        "deepspeed_stage_2_offload",
        "deepspeed_stage_3",
        "deepspeed_stage_3_offload",
    ],
)
def test_resolve_resume_checkpoint_path_accepts_all_deepspeed_strategies(strategy: str, tmp_path: Path):
    ckpt_dir = _make_ds_checkpoint_dir(tmp_path, "rwkv-11.pth", 100)
    assert train.resolve_resume_checkpoint_path(str(ckpt_dir), strategy) == str(ckpt_dir)


def test_resolve_resume_checkpoint_path_rejects_non_deepspeed_strategy(tmp_path: Path):
    ckpt_dir = _make_ds_checkpoint_dir(tmp_path, "rwkv-11.pth", 100)
    with pytest.raises(ValueError, match="DeepSpeed sharded checkpoint"):
        train.resolve_resume_checkpoint_path(str(ckpt_dir), "ddp")


def test_prune_old_checkpoints_keeps_latest_numbered_entries_and_preserves_non_numbered(tmp_path: Path):
    _make_ds_checkpoint_dir(tmp_path, "rwkv-init.pth", 100)
    _make_ds_checkpoint_dir(tmp_path, "rwkv-1.pth", 110)
    _make_ds_checkpoint_dir(tmp_path, "rwkv-2.pth", 120)
    _make_file_checkpoint(tmp_path, "rwkv-3.pth", 130)
    _make_file_checkpoint(tmp_path, "rwkv-final.pth", 140)

    args = SimpleNamespace(proj_dir=str(tmp_path), keep_last_n_checkpoints=2)
    trainer_mod.prune_old_checkpoints(args)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "rwkv-2.pth",
        "rwkv-3.pth",
        "rwkv-final.pth",
        "rwkv-init.pth",
    ]


@pytest.mark.parametrize(
    "strategy",
    [
        "deepspeed_stage_1",
        "deepspeed_stage_2",
        "deepspeed_stage_2_offload",
        "deepspeed_stage_3",
        "deepspeed_stage_3_offload",
    ],
)
def test_my_save_uses_trainer_save_checkpoint_for_all_deepspeed_strategies(strategy: str):
    calls = []

    class DummyTrainer:
        def save_checkpoint(self, path, weights_only=True):
            calls.append((path, weights_only))

    trainer_mod.my_save(SimpleNamespace(strategy=strategy), DummyTrainer(), {"x": 1}, "dummy.pth")
    assert calls == [("dummy.pth", False)]


def test_my_save_uses_torch_save_for_non_deepspeed(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_torch_save(payload, path):
        calls.append((payload, path))

    monkeypatch.setattr(trainer_mod.torch, "save", fake_torch_save)
    trainer_mod.my_save(SimpleNamespace(strategy="ddp"), object(), {"x": 1}, "dummy.pth")
    assert calls == [({"x": 1}, "dummy.pth")]


def test_save_train_checkpoint_prunes_old_deepspeed_directories_and_waits_for_barriers(tmp_path: Path):
    _make_ds_checkpoint_dir(tmp_path, "rwkv-1.pth", 100)

    barrier_calls = []
    save_calls = []

    class DummyStrategy:
        def barrier(self):
            barrier_calls.append("barrier")

    class DummyTrainer:
        def __init__(self):
            self.strategy = DummyStrategy()
            self.is_global_zero = True

        def save_checkpoint(self, path, weights_only=True):
            save_calls.append((path, weights_only))
            ckpt_dir = Path(path)
            ckpt_dir.mkdir()
            (ckpt_dir / "latest").write_text("checkpoint", encoding="utf-8")
            (ckpt_dir / "zero_pp_rank_0_mp_rank_00_model_states.pt").write_text("", encoding="utf-8")

    args = SimpleNamespace(
        strategy="deepspeed_stage_2",
        proj_dir=str(tmp_path),
        keep_last_n_checkpoints=1,
        data_type="binidx",
    )

    trainer_mod.save_train_checkpoint(args, DummyTrainer(), SimpleNamespace(state_dict=lambda: {"x": 1}), str(tmp_path / "rwkv-2.pth"))

    assert save_calls == [(str(tmp_path / "rwkv-2.pth"), False)]
    assert barrier_calls == ["barrier", "barrier"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rwkv-2.pth"]


def test_save_train_checkpoint_marks_inflight_batch_completed_during_save(monkeypatch: pytest.MonkeyPatch):
    seen_progress = []

    def fake_my_save(args, trainer, dd, ff):
        progress = trainer.fit_loop.epoch_loop.batch_progress
        seen_progress.append(
            (
                progress.current.ready,
                progress.current.processed,
                progress.current.completed,
                progress.total.ready,
                progress.total.processed,
                progress.total.completed,
            )
        )

    monkeypatch.setattr(trainer_mod, "my_save", fake_my_save)

    current = SimpleNamespace(ready=20, started=20, processed=20, completed=19)
    total = SimpleNamespace(ready=20, started=20, processed=20, completed=19)
    trainer = SimpleNamespace(
        is_global_zero=False,
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(batch_progress=SimpleNamespace(current=current, total=total))),
    )
    args = SimpleNamespace(strategy="", data_type="binidx", proj_dir="", keep_last_n_checkpoints=0)

    trainer_mod.save_train_checkpoint(args, trainer, SimpleNamespace(state_dict=lambda: {"x": 1}), "dummy.pth")

    assert seen_progress == [(20, 20, 20, 20, 20, 20)]
    assert current.completed == 19
    assert total.completed == 19


def test_progress_snapshot_handles_missing_batch_progress():
    trainer = SimpleNamespace()
    assert trainer_mod.mark_current_batch_completed_for_checkpoint(trainer) == []
    trainer_mod.restore_progress_snapshot([])


def test_trainer_helper_branches_cover_save_prune_move_and_grad_norm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = SimpleNamespace(
        state_dict=lambda: {
            "encoder.weight": torch.tensor([1]),
            "decoder.weight": torch.tensor([2]),
            "head.weight": torch.tensor([3]),
        }
    )
    assert sorted(trainer_mod.build_save_dict(SimpleNamespace(data_type="wds_img"), module)) == [
        "decoder.weight",
        "encoder.weight",
    ]
    assert "head.weight" in trainer_mod.build_save_dict(SimpleNamespace(data_type="binidx"), module)

    trainer_mod.prune_old_checkpoints(SimpleNamespace(proj_dir=str(tmp_path / "missing"), keep_last_n_checkpoints=2))
    trainer_mod.prune_old_checkpoints(SimpleNamespace(proj_dir=str(tmp_path), keep_last_n_checkpoints=0))
    _make_file_checkpoint(tmp_path, "rwkv-1.pth", 100)
    _make_file_checkpoint(tmp_path, "rwkv-2.pth", 200)
    trainer_mod.prune_old_checkpoints(SimpleNamespace(proj_dir=str(tmp_path), keep_last_n_checkpoints=1))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rwkv-2.pth"]

    tensor = torch.tensor([1])
    moved = trainer_mod.move_batch_to_device(
        {"a": (tensor,), "b": [tensor], "c": "unchanged"},
        torch.device("cpu"),
    )
    assert torch.equal(moved["a"][0], tensor)
    assert torch.equal(moved["b"][0], tensor)
    assert moved["c"] == "unchanged"

    trainer_mod.strategy_barrier(SimpleNamespace(strategy=SimpleNamespace(barrier=lambda: (_ for _ in ()).throw(RuntimeError("boom")))))

    class OwnerWithBadNorm:
        def get_global_grad_norm(self):
            raise TypeError("try next")

        def get_grad_norm(self):
            return "not-float"

        gradient_norm = None

    assert trainer_mod.get_global_grad_norm(SimpleNamespace(strategy=SimpleNamespace(model=OwnerWithBadNorm())), object()) is None
    assert trainer_mod.get_global_grad_norm(SimpleNamespace(strategy=""), SimpleNamespace(parameters=None)) is None

    module_without_grads = torch.nn.Linear(2, 1)
    assert trainer_mod.get_global_grad_norm(SimpleNamespace(strategy=""), module_without_grads) is None


def test_train_callback_logging_state_can_initialize_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    init_calls = []
    fake_wandb = SimpleNamespace(init=lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    callback = trainer_mod.train_callback(
        SimpleNamespace(
            proj_dir=str(tmp_path),
            wandb="project",
            run_name="wandb-test",
            my_timestamp="2026-06-23-11-45-00",
        )
    )
    callback._ensure_run_logging_state(SimpleNamespace(is_global_zero=False))
    assert not (tmp_path / "train_log.txt").exists()

    trainer = SimpleNamespace(global_step=7, is_global_zero=True, strategy=SimpleNamespace(config={"x": 1}))
    callback._ensure_run_logging_state(trainer)
    trainer.my_log.close()

    assert init_calls == [
        {
            "project": "project",
            "name": "wandb-test 2026-06-23-11-45-00",
            "config": callback.args,
            "save_code": False,
        }
    ]
    assert trainer.my_wandb is fake_wandb
    assert "RESUME RUN @ step 7" in (tmp_path / "train_log.txt").read_text(encoding="utf-8")


def test_train_callback_final_checkpoint_and_epoch_end_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events = []

    def fake_save(args, trainer, pl_module, file_name):
        events.append(Path(file_name).name)

    monkeypatch.setattr(trainer_mod, "save_train_checkpoint", fake_save)

    args = SimpleNamespace(
        data_type="sft_binidx",
        strategy="",
        proj_dir=str(tmp_path),
        magic_prime=12,
        save_every_n_steps=0,
        save_at_step=0,
        sft_eval_every_n_steps=0,
        ctx_len=16,
        real_bsz=2,
        effective_bsz=2,
        epoch_begin=0,
        epoch_steps=100,
        warmup_steps=0,
        my_exit_tokens=0,
        lr_init=1e-4,
        lr_final=1e-5,
        lr_wsd_decay_iters=0,
        lr_wsd_decay_style="cosine",
        weight_decay=0.0,
        wandb="",
        run_name="final-test",
        my_timestamp="2026-06-23-11-50-00",
        epoch_save=1,
        epoch_count=2,
    )
    callback = trainer_mod.train_callback(args)
    callback.log = lambda *args, **kwargs: None

    trainer = SimpleNamespace(
        global_step=4,
        current_epoch=1,
        is_global_zero=True,
        strategy=SimpleNamespace(config={}),
        optimizers=[SimpleNamespace(param_groups=[{"weight_decay": 0.0, "my_lr_scale": 1.0}])],
        my_loss_all=torch.tensor([1.0]),
    )
    callback.on_train_batch_start(trainer, object(), None, 0)
    trainer.global_step = 5
    callback.on_train_batch_end(trainer, object(), None, None, 0)
    callback.on_train_epoch_end(trainer, object())
    trainer.my_log.close()

    assert events == ["rwkv-final.pth", "rwkv-1.pth"]


def test_train_callback_eval_zero_mask_and_invalid_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(trainer_mod, "strategy_barrier", lambda trainer: None)

    class FakeModule:
        device = torch.device("cpu")
        training = False

        def eval(self):
            self.training = False

        def training_step(self, batch, batch_idx):
            return torch.tensor(9.0)

    class FakeWandb:
        def __init__(self):
            self.records = []

        def log(self, values, step):
            self.records.append((values, step))

    class ZeroMaskLoader:
        dataset = SimpleNamespace()

        def __iter__(self):
            return iter(
                [
                    (
                        torch.zeros((1, 2), dtype=torch.long),
                        torch.zeros((1, 2), dtype=torch.long),
                        torch.zeros((1, 2)),
                    )
                ]
            )

    callback = trainer_mod.train_callback(
        SimpleNamespace(proj_dir=str(tmp_path), wandb="enabled", run_name="eval-zero", my_timestamp="now", sft_eval_steps=0),
        eval_loader=ZeroMaskLoader(),
    )
    trainer = SimpleNamespace(
        global_rank=0,
        world_size=1,
        is_global_zero=True,
        strategy=SimpleNamespace(barrier=lambda: None),
        my_log=open(tmp_path / "train_log.txt", "a"),
        my_wandb=FakeWandb(),
    )
    callback._run_sft_eval(trainer, FakeModule(), real_step=3)
    assert trainer.my_wandb.records[0][0]["eval/loss"] == 0.0
    assert trainer.my_wandb.records[0][0]["eval/ppl"] == 1.0

    class BadLoader:
        dataset = SimpleNamespace()

        def __iter__(self):
            return iter([(torch.zeros((1, 2), dtype=torch.long),)])

    callback.eval_loader = BadLoader()
    with pytest.raises(ValueError, match="SFT eval requires"):
        callback._run_sft_eval(trainer, FakeModule(), real_step=4)
    trainer.my_log.close()


def test_generate_init_weight_stage0_and_stage1_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    save_calls = []

    def fake_save(payload, path):
        save_calls.append((payload, path))

    monkeypatch.setattr(trainer_mod.torch, "save", fake_save)

    trainer_mod.generate_init_weight(
        SimpleNamespace(args=SimpleNamespace(train_stage=0), generate_init_weight=lambda: {"x": torch.tensor([1.0])}),
        str(tmp_path / "init.pth"),
    )
    assert save_calls[-1][1] == str(tmp_path / "init.pth")

    monkeypatch.setattr(trainer_mod.torch, "load", lambda path, map_location=None: {"x": torch.tensor([0.0, 2.0])})
    monkeypatch.setattr(trainer_mod, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)), raising=False)

    with pytest.raises(SystemExit):
        trainer_mod.generate_init_weight(
            SimpleNamespace(
                args=SimpleNamespace(train_stage=1, load_model="base.pth"),
                generate_init_weight=lambda: {"x": torch.zeros(4)},
            ),
            str(tmp_path / "stage1.pth"),
        )
    interpolated = save_calls[-1][0]["x"]
    assert torch.equal(interpolated, torch.tensor([0.0, 1.0, 2.0, 2.0]))


def test_resume_smoke_defaults_require_real_restore_and_progress_markers():
    assert resume_smoke.default_require_patterns() == [
        "Resuming trainer state from",
        "Restoring states from the checkpoint path at",
    ]
    assert "Epoch " in resume_smoke.default_progress_patterns()
    assert resume_smoke.line_has_failure_marker("KeyError: missing lr_schedulers")


def test_train_callback_initializes_logging_state_when_resuming_mid_run(tmp_path: Path):
    callback = trainer_mod.train_callback(
        SimpleNamespace(
            data_type="binidx",
            strategy="deepspeed_stage_3_offload",
            proj_dir=str(tmp_path),
            wandb="",
            my_timestamp="2026-06-02-17-00-00",
            run_name="resume-test",
            epoch_begin=0,
            epoch_steps=5040,
            warmup_steps=10,
            my_exit_tokens=1000,
            ctx_len=16,
            real_bsz=8,
            lr_init=1e-4,
            lr_final=1e-5,
            lr_wsd_decay_iters=0,
            lr_wsd_decay_style="cosine",
            weight_decay=0.01,
            magic_prime=0,
            save_every_n_steps=0,
            save_at_step=0,
        )
    )
    callback.log = lambda *args, **kwargs: None

    trainer = SimpleNamespace(
        global_step=50,
        is_global_zero=True,
        strategy=SimpleNamespace(config={"zero_optimization": {"stage": 3}}),
        optimizers=[SimpleNamespace(param_groups=[{"weight_decay": 0.01, "my_lr_scale": 1.0}])],
        my_loss_all=torch.tensor([1.25], dtype=torch.float32),
    )

    callback.on_train_batch_start(trainer, object(), None, 0)
    callback.on_train_batch_end(trainer, object(), None, None, 0)

    assert trainer.my_loss_count == 1
    assert trainer.my_loss_sum == pytest.approx(1.25)
    assert (tmp_path / "train_log.txt").exists()
    trainer.my_log.close()


def test_train_callback_sets_dataset_step_offset_from_restored_global_step():
    callback = trainer_mod.train_callback(
        SimpleNamespace(
            epoch_begin=0,
            epoch_steps=5040,
        )
    )

    class MyDataset:
        pass

    dataset = MyDataset()
    trainer = SimpleNamespace(
        global_rank=3,
        current_epoch=2,
        world_size=8,
        global_step=2 * 5040 + 50,
        is_global_zero=False,
        train_dataloader=SimpleNamespace(dataset=SimpleNamespace(datasets=dataset)),
    )

    callback.on_train_epoch_start(trainer, object())

    assert dataset.global_rank == 3
    assert dataset.real_epoch == 2
    assert dataset.world_size == 8
    assert dataset.step_offset == 50


def test_dataset_getitem_applies_resume_step_offset():
    dataset = object.__new__(dataset_mod.MyDataset)
    dataset.args = SimpleNamespace(ctx_len=4, magic_prime=11, micro_bsz=2)
    dataset.global_rank = 1
    dataset.real_epoch = 0
    dataset.world_size = 8
    dataset.samples_per_epoch = 40320
    dataset.step_offset = 5

    class DummyData:
        def __init__(self):
            self.calls = []

        def get(self, idx, offset, length):
            self.calls.append((idx, offset, length))
            return np.arange(offset, offset + length, dtype=np.int64)

    dataset.data = DummyData()

    x, y = dataset_mod.MyDataset.__getitem__(dataset, 0)

    logical_idx = 0 + dataset.step_offset * dataset.args.micro_bsz
    ii = 1 + dataset.real_epoch * dataset.samples_per_epoch + (logical_idx * dataset.world_size) + dataset.global_rank
    factor = int(dataset.args.magic_prime * ((math.sqrt(5) - 1) / 2))
    expected_offset = ((factor * ii * ii * ii) % dataset.args.magic_prime) * dataset.args.ctx_len

    assert dataset.data.calls == [(0, expected_offset, dataset.args.ctx_len + 1)]
    assert torch.equal(x, torch.tensor([expected_offset + i for i in range(dataset.args.ctx_len)], dtype=torch.long))
    assert torch.equal(y, torch.tensor([expected_offset + i for i in range(1, dataset.args.ctx_len + 1)], dtype=torch.long))


def test_dataset_initializes_resume_position_before_first_epoch(monkeypatch: pytest.MonkeyPatch):
    class DummyIndex:
        _dtype_size = 2

    class DummyDataset:
        def __init__(self, path):
            self._bin_buffer = bytes(4096)
            self._index = DummyIndex()

    monkeypatch.setattr(dataset_mod, "MMapIndexedDataset", DummyDataset)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    dataset = dataset_mod.MyDataset(
        SimpleNamespace(
            vocab_size=65536,
            data_file="dummy",
            epoch_steps=5040,
            real_bsz=8,
            train_stage=0,
            ctx_len=4,
            magic_prime=509,
            epoch_begin=0,
            resume_epoch=2,
            resume_step_offset=17,
        )
    )

    assert dataset.global_rank == 3
    assert dataset.world_size == 8
    assert dataset.real_epoch == 2
    assert dataset.step_offset == 17

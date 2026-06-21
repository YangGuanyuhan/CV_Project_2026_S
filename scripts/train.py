import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "work_dirs" / "gdino_ft_short" / "grounding_dino_swin-t_finetune_16xb2_1x_coco.py"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "coco"
DEFAULT_BERT = PROJECT_ROOT / "models" / "bert-base-uncased"
DEFAULT_PRETRAINED = PROJECT_ROOT / "checkpoints" / "groundingdino_swint_ogc_mmdet-822d7e9d.pth"
DEFAULT_WORK_DIR = PROJECT_ROOT / "work_dirs" / "gdino_ft_reproduce"


def resolve_bert_name(value: str) -> str:
    """Use a local BERT directory only when it contains real model files."""
    path = Path(value).expanduser()
    if path.exists():
        if path.is_dir() and (path / "config.json").exists():
            return str(path.resolve())
        print(
            f"[WARN] BERT path exists but is incomplete: {path}. "
            "Falling back to HuggingFace model name 'bert-base-uncased'."
        )
        return "bert-base-uncased"
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Grounding DINO with MMDetection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="MMDetection config file. The default is the dumped config used for our 6-epoch run.",
    )
    parser.add_argument(
        "--mmdet-root",
        type=str,
        default=os.environ.get("MMDET_ROOT", str(PROJECT_ROOT.parent / "mmdetection")),
        help="Path to an MMDetection checkout containing tools/train.py.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=str(DEFAULT_WORK_DIR),
        help="Directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=str(DEFAULT_PRETRAINED),
        help="MMDetection-format Grounding DINO checkpoint to initialize from.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="COCO root directory containing train2017, val2017 and annotations.",
    )
    parser.add_argument(
        "--bert",
        type=str,
        default=str(DEFAULT_BERT),
        help="Local bert-base-uncased path or HuggingFace model name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint to resume training from. If omitted, load_from is used for initialization.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the MMDetection command without running it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mmdet_root = Path(args.mmdet_root).expanduser().resolve()
    train_py = mmdet_root / "tools" / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(
            f"MMDetection train.py not found: {train_py}. "
            "Set --mmdet-root or MMDET_ROOT to your mmdetection checkout."
        )

    config = Path(args.config).expanduser().resolve()
    if not config.exists():
        raise FileNotFoundError(f"Config file not found: {config}")

    data_root = str(Path(args.data_root).expanduser().resolve()) + "/"
    work_dir = str(Path(args.work_dir).expanduser().resolve())
    pretrained = str(Path(args.pretrained).expanduser().resolve())
    bert_name = resolve_bert_name(args.bert)

    cfg_options = [
        f"work_dir={work_dir}",
        f"data_root={data_root}",
        f"lang_model_name={bert_name}",
        f"model.language_model.name={bert_name}",
        f"train_cfg.max_epochs={args.epochs}",
        f"train_dataloader.batch_size={args.batch_size}",
        f"optim_wrapper.optimizer.lr={args.lr}",
        f"train_dataloader.dataset.data_root={data_root}",
        f"val_dataloader.dataset.data_root={data_root}",
        f"test_dataloader.dataset.data_root={data_root}",
        f"val_evaluator.ann_file={data_root}annotations/instances_val2017.json",
        f"test_evaluator.ann_file={data_root}annotations/instances_val2017.json",
    ]

    if args.resume:
        cfg_options.extend(["resume=True", f"load_from={Path(args.resume).expanduser().resolve()}"])
    else:
        cfg_options.extend(["resume=False", f"load_from={pretrained}"])

    cmd = [
        sys.executable,
        str(train_py),
        str(config),
        "--work-dir",
        work_dir,
        "--cfg-options",
        *cfg_options,
    ]

    print("Running MMDetection fine-tuning command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))

    if args.dry_run:
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{mmdet_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()

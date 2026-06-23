"""Copy base model and OpenBookQA snapshots into this experiment directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "parametric-faithfulness-ToT"


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        print(f"Already present: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    print(f"Copied {source} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    args = parser.parse_args()
    copy_tree(
        args.source_root / "local_models" / "Llama-3.2-3B-Instruct",
        ROOT / "local_models" / "Llama-3.2-3B-Instruct",
    )
    copy_tree(args.source_root / "local_datasets" / "openbookqa", ROOT / "local_datasets" / "openbookqa")


if __name__ == "__main__":
    main()

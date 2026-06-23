"""ToT FUR entry point using a JSONL cache produced by tot_generation.py."""

import sys

from unlearn import main


def add_default(flag, value=None):
    if flag not in sys.argv:
        sys.argv.append(flag)
        if value is not None:
            sys.argv.append(value)


if __name__ == "__main__":
    union = "--union" in sys.argv
    if union:
        sys.argv.remove("--union")
    add_default("--source_tag", "tot_union" if union else "tot_selected")
    add_default("--result_root", "final_result_ToT_union" if union else "final_result_ToT")
    add_default("--strategy", "tree_union" if union else "sentencize")
    if union and "--no-stepwise" not in sys.argv:
        sys.argv.append("--no-stepwise")
    main()

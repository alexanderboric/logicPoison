import argparse
import os
from typing import List

from src.global_poison import run_corpus
from src.logic_poison import run_poison
from src.query_centric import run_queries

STAGE_ORDER = ["global", "query", "logic"]


def list_datasets(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        ds
        for ds in os.listdir(root)
        if os.path.isdir(os.path.join(root, ds))
    )


def pick_stages(stgs: List[str]) -> List[str]:
    if "all" in stgs:
        return STAGE_ORDER
    sel = set(stgs)
    return [s for s in STAGE_ORDER if s in sel]


def pick_datasets(ds_args: List[str], avail: List[str]) -> List[str]:
    if "all" in ds_args:
        return avail

    bad = [ds for ds in ds_args if ds not in avail]
    if bad:
        raise ValueError(f"Unknown datasets under data_root: {bad}")
    return ds_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single entry for logic-poison pipeline: global/query/logic stages."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "global", "query", "logic"],
        help='Choose stages, e.g. "--stages global logic" or "--stages all".',
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help='Choose datasets, e.g. "--datasets hotpotqa musique" or "--datasets all".',
    )
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--corpus_entities_root", type=str, default="results/corpus_entities")
    parser.add_argument("--queries_entities_root", type=str, default="results/queries_entities")
    parser.add_argument("--poisoned_root", type=str, default="results/poisoned_data")

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--query_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--queue_factor", type=int, default=4)

    parser.add_argument("--top_ratio", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    available = list_datasets(args.data_root)
    if not available:
        print(f"[WARN] No datasets found under {args.data_root}")
        return

    selected_datasets = pick_datasets(args.datasets, available)

    selected_stages = pick_stages(args.stages)
    print(f"[INFO] Selected stages  : {selected_stages}")
    print(f"[INFO] Selected datasets: {selected_datasets}")

    if "global" in selected_stages:
        print("\n[PIPELINE] Stage global: corpus entity statistics")
        run_corpus(
            selected_datasets,
            data_root=args.data_root,
            output_root=args.corpus_entities_root,
            batch_size=args.batch_size,
        )

    if "query" in selected_stages:
        print("\n[PIPELINE] Stage query: query-centric entity extraction")
        run_queries(
            selected_datasets,
            data_root=args.data_root,
            output_dir=args.queries_entities_root,
            max_workers=args.max_workers,
            queue_factor=args.queue_factor,
            model=args.query_model,
        )

    if "logic" in selected_stages:
        print("\n[PIPELINE] Stage logic: final corpus poisoning")
        run_poison(
            selected_datasets,
            data_root=args.data_root,
            corpus_stats_root=args.corpus_entities_root,
            queries_entities_root=args.queries_entities_root,
            poisoned_root=args.poisoned_root,
            top_ratio=args.top_ratio,
        )

    print("\n[INFO] Pipeline finished.")


if __name__ == "__main__":
    main()

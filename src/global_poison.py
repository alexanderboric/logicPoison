import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

import spacy
from tqdm import tqdm

if spacy.prefer_gpu():
    print("[INFO] Using GPU for spaCy")
else:
    print("[INFO] GPU not available, using CPU")

nlp = spacy.load("en_core_web_trf")  # MUST load after prefer_gpu()


def corpus_stats(
    input_path: str,
    batch_size: int = 32,
) -> Dict[str, Counter]:
    """
    Process corpus.jsonl using spaCy + GPU + nlp.pipe in batches.
    Returns:
        { label: Counter({ entity_text: count }) }
    """
    cnts: Dict[str, Counter] = defaultdict(Counter)

    def txt_stream():
        with open(input_path, "r", encoding="utf-8") as fin:
            for ln in fin:
                ln = ln.strip()
                if not ln:
                    continue
                rec = json.loads(ln)
                txt = rec.get("text", "")
                txt = txt.replace("(", " ( ").replace(")", " ) ")
                yield txt

    with open(input_path, "r", encoding="utf-8") as f:
        n_docs = sum(1 for _ in f)
    print(f"[INFO] corpus size: {n_docs} docs")

    for doc in tqdm(
        nlp.pipe(txt_stream(), batch_size=batch_size, n_process=1),
        total=n_docs,
        desc=f"NER (GPU) on {os.path.basename(input_path)}",
        unit="doc",
    ):
        for ent in doc.ents:
            cnts[ent.label_][ent.text] += 1

    return cnts


def save_stats(stats: Dict[str, Counter], output_path: str):
    data = {
        label: sorted(counter.items(), key=lambda x: -x[1])
        for label, counter in stats.items()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved corpus entity stats to: {output_path}")


def run_corpus(
    dataset_names: Iterable[str],
    data_root: str = "datasets",
    output_root: str = "results/corpus_entities",
    batch_size: int = 32,
):
    os.makedirs(output_root, exist_ok=True)

    for ds in dataset_names:
        in_path = os.path.join(data_root, ds, "corpus.jsonl")
        out_path = os.path.join(output_root, f"{ds}.json")

        print(f"\n[INFO] Processing corpus for dataset: {ds}")
        print(f"       Input : {in_path}")
        print(f"       Output: {out_path}")

        cnts = corpus_stats(
            in_path,
            batch_size=batch_size,
        )
        save_stats(cnts, out_path)


def run_all(
    data_root: str = "datasets",
    output_root: str = "results/corpus_entities",
    batch_size: int = 32,
):
    pat = os.path.join(data_root, "*/corpus.jsonl")
    files = sorted(glob.glob(pat))
    ds_list = [os.path.basename(os.path.dirname(p)) for p in files]

    if not ds_list:
        print("[WARN] No corpus.jsonl files found.")
        return

    run_corpus(
        ds_list,
        data_root=data_root,
        output_root=output_root,
        batch_size=batch_size,
    )


def list_datasets(data_root: str) -> List[str]:
    pat = os.path.join(data_root, "*/corpus.jsonl")
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(pat))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract corpus entity statistics by dataset.")
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--output_root", type=str, default="results/corpus_entities")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help='Dataset names, e.g. "hotpotqa musique", or "all".',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    available = list_datasets(args.data_root)
    if not available:
        print(f"[WARN] No datasets found under {args.data_root}")
        return

    if "all" in args.datasets:
        selected = available
    else:
        selected = args.datasets

    run_corpus(
        selected,
        data_root=args.data_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

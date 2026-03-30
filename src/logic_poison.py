import argparse
import os
import json
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm


# ==============================
# 1. Load corpus entity statistics
# ==============================

def load_corpus_stats(stats_path: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    stats_path: results/corpus_entities/hotpotqa.json
    Returns: {label: [(entity, count), ...]} already sorted by frequency
    """
    with open(stats_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # data: {label: [[ent, cnt], ...]}
    out: Dict[str, List[Tuple[str, int]]] = {}
    for lbl, rows in raw.items():
        out[lbl] = [(ent, int(cnt)) for ent, cnt in rows]
    return out


def build_corpus_pools(
    stats: Dict[str, List[Tuple[str, int]]],
    top_ratio: float = 0.1,
) -> Tuple[Dict[str, List[str]], Dict[str, set]]:
    """
    Input corpus statistics, returns:
      - pools: Top top_ratio frequency entity list for each type, used as replacement candidates, in original frequency order
      - all_sets: Full set for each type, used to determine if entities in query have appeared in corpus
    """
    pools: Dict[str, List[str]] = {}
    all_set: Dict[str, set] = {}

    for lbl, rows in stats.items():
        all_ents = [ent for ent, _ in rows]
        all_set[lbl] = set(all_ents)

        if not rows:
            pools[lbl] = []
            continue

        k = int(len(rows) * top_ratio)
        if k < 1:
            k = 1
        topk = [ent for ent, _ in rows[:k]]  # Frequency from high to low
        pools[lbl] = topk

    return pools, all_set


# ==============================
# 2. Collect entities to be replaced from queries_entities (preserve order)
# ==============================

def load_query_entities(
    queries_entities_path: str,
    corpus_all_sets: Dict[str, set],
) -> Dict[str, List[str]]:
    """
    Read entities from results/queries_entities/{dataset}.jsonl,
    only keep those:
      - type is in corpus_all_sets
      - entity text has appeared in corpus_all_sets[type]

    Returns: {type: [entities_to_replace_in_order]}
    For the same entity, only keep the first occurrence position to ensure stable order.
    """
    atk: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, set] = defaultdict(set)

    with open(queries_entities_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ents = rec.get("entities") or []
            for ent in ents:
                typ = ent.get("type")
                txt = ent.get("entity", "").strip()
                if not txt or not typ:
                    continue
                # Only consider entities that have appeared in corpus
                if typ in corpus_all_sets and txt in corpus_all_sets[typ]:
                    if txt not in seen[typ]:
                        seen[typ].add(txt)
                        atk[typ].append(txt)

    return atk


# ==============================
# 3. Build circular replacement mapping + statistics
# ==============================

def build_replace_map(
    attack_entities: Dict[str, List[str]],
    corpus_pools: Dict[str, List[str]],
    label2count: Dict[str, Dict[str, int]],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    attack_entities: {type: [entities from queries (in order), each appears in corpus]}
    corpus_pools   : {type: [top-5%-by-frequency entities from corpus, in freq order]}
    label2count    : {label: {entity: count}}

    For each type, construct a pool in the following order:
        pool(type) = [top-5% (in frequency order)] + [entities from query (in occurrence order), deduplicated]

    Then perform "reverse circular permutation":
        e0 -> e_{n-1}
        e1 -> e0
        e2 -> e1
        ...
        e_{n-1} -> e_{n-2}

    Returns:
      - replace_map: {src_entity: tgt_entity}
      - poison_stats: {
            type: {
                "num_entities": int,   # Number of entities participating in circular replacement
                "total_freq": int      # Total frequency of these entities in original corpus
            }, ...
        }
    """
    rmap: Dict[str, str] = {}
    pstat: Dict[str, Dict[str, int]] = {}

    for typ, q_list in attack_entities.items():
        top = corpus_pools.get(typ, [])  # Already sorted by frequency descending

        # First add corpus top-5% (preserve original order)
        pool: List[str] = list(top)

        # Then append by query occurrence order, avoiding duplicates
        for ent in q_list:
            if ent not in pool:
                pool.append(ent)

        n = len(pool)
        if n < 2:
            continue

        # Statistics: number of entities participating in circular replacement + total frequency of these entities in corpus
        cnt = label2count.get(typ, {})
        freq = sum(cnt.get(ent, 0) for ent in pool)
        pstat[typ] = {
            "num_entities": n,
            "total_freq": int(freq),
        }

        # Reverse circular permutation:
        #   First -> Last
        #   Each other -> Previous one
        # This is still a permutation, avoiding the problem of replacing the same entity multiple times
        for i, src in enumerate(pool):
            if i == 0:
                tgt = pool[-1]
            else:
                tgt = pool[i - 1]
            rmap[src] = tgt

    return rmap, pstat


# ==============================
# 4. Perform text replacement in corpus.jsonl (single regex, avoid cascading replacement)
# ==============================

def poison_corpus_file(
    corpus_path: str,
    output_path: str,
    replace_map: Dict[str, str],
):
    """
    Read datasets/{dataset}/corpus.jsonl,
    perform text replacement on text field based on replace_map,
    output to output_path (e.g., results/poisoned_data/{dataset}/corpus.jsonl)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not replace_map:
        # If there's no replacement, just copy the original file
        with open(corpus_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)
        return

    # Construct unified regex to match all entities to be replaced at once, avoiding multiple cascading replacements
    keys = list(replace_map.keys())
    # Sorting from long to short can be slightly safer in pattern matching (avoid long entities being preempted by short ones)
    keys_sorted = sorted(keys, key=len, reverse=True)
    pattern = r'(?<!\w)(' + '|'.join(map(re.escape, keys_sorted)) + r')(?!\w)'
    regex = re.compile(pattern)

    def repl(m: re.Match) -> str:
        src = m.group(1)
        return replace_map.get(src, src)

    with open(corpus_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"Poisoning {os.path.basename(corpus_path)}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")

            # Single scan replaces all keys, no cascading replacement will occur
            text = regex.sub(repl, text)

            obj["text"] = text
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ==============================
# 5. Complete pipeline for a single dataset
# ==============================

def logicPoison(
    dataset_name: str,
    data_root: str = "datasets",
    corpus_stats_root: str = "results/corpus_entities",
    queries_entities_root: str = "results/queries_entities",
    poisoned_root: str = "results/poisoned_data",
    top_ratio: float = 0.05,
):
    """
    For a single dataset (e.g., 'hotpotqa'):
      1) Read corpus entity statistics, build top top_ratio replacement pool
      2) Filter entities from queries_entities that have appeared in corpus (preserve order)
      3) Construct circular replacement mapping + statistics
      4) Perform text replacement on corpus.jsonl, output to results/poisoned_data/{dataset}/corpus.jsonl
      5) Copy queries.jsonl and qrels/* to build BEIR-compatible directory
      6) Write replacement statistics to results/poisoned_data/{dataset}/poison_stats_{dataset}.json
    """

    # Original dataset directory: datasets/{dataset_name}
    orig_dir = os.path.join(data_root, dataset_name)
    orig_corpus_path = os.path.join(orig_dir, "corpus.jsonl")
    orig_queries_path = os.path.join(orig_dir, "queries.jsonl")
    orig_answers_path = os.path.join(orig_dir, "answers.jsonl")
    orig_qrels_dir = os.path.join(orig_dir, "qrels")

    # Poisoned dataset directory: results/poisoned_data/{dataset_name}
    poisoned_dir = os.path.join(poisoned_root, dataset_name)
    poisoned_corpus_path = os.path.join(poisoned_dir, "corpus.jsonl")

    corpus_stats_path = os.path.join(corpus_stats_root, f"{dataset_name}.json")
    queries_entities_path = os.path.join(queries_entities_root, f"{dataset_name}.jsonl")

    print(f"\n[INFO] === Dataset: {dataset_name} ===")
    print(f"[INFO] corpus stats        : {corpus_stats_path}")
    print(f"[INFO] queries entities    : {queries_entities_path}")
    print(f"[INFO] original dataset dir: {orig_dir}")
    print(f"[INFO] poisoned dataset dir: {poisoned_dir}")

    # 1) Corpus entity statistics (sorted by frequency)
    stats = load_corpus_stats(corpus_stats_path)
    corpus_pools, corpus_all_sets = build_corpus_pools(stats, top_ratio=top_ratio)

    # For convenience in statistics, convert stats to label -> {ent: count}
    label2count: Dict[str, Dict[str, int]] = {
        label: {ent: cnt for ent, cnt in lst}
        for label, lst in stats.items()
    }

    # 2) Entities to be replaced in query (and have appeared in corpus), in occurrence order
    attack_entities = load_query_entities(queries_entities_path, corpus_all_sets)
    # For ablation: uncomment to disable query entity injection
    # attack_entities = {etype: [] for etype in corpus_pools.keys()}

    # 3) Construct circular replacement mapping + statistics
    replace_map, poison_stats = build_replace_map(
        attack_entities, corpus_pools, label2count
    )

    total_src = sum(v["num_entities"] for v in poison_stats.values())
    total_freq = sum(v["total_freq"] for v in poison_stats.values())
    print(f"[INFO] Total types poisoned       : {len(poison_stats)}")
    print(f"[INFO] Total unique entities      : {total_src}")
    print(f"[INFO] Total occurrences in corpus: {total_freq}")

    # Ensure target directory exists
    os.makedirs(poisoned_dir, exist_ok=True)

    # 4) Perform replacement on corpus, write to results/poisoned_data/{dataset}/corpus.jsonl
    poison_corpus_file(orig_corpus_path, poisoned_corpus_path, replace_map)

    # 5) Copy original queries.jsonl and qrels/* to build BEIR-compatible directory
    if os.path.exists(orig_queries_path):
        shutil.copyfile(orig_queries_path, os.path.join(poisoned_dir, "queries.jsonl"))
    else:
        print(f"[WARN] queries.jsonl not found in {orig_dir}")

    if os.path.exists(orig_answers_path):
        shutil.copyfile(orig_answers_path, os.path.join(poisoned_dir, "answers.jsonl"))
    else:
        print(f"[WARN] answers.jsonl not found in {orig_dir}")
        
    if os.path.isdir(orig_qrels_dir):
        poisoned_qrels_dir = os.path.join(poisoned_dir, "qrels")
        shutil.copytree(orig_qrels_dir, poisoned_qrels_dir, dirs_exist_ok=True)
    else:
        print(f"[WARN] qrels dir not found in {orig_dir}")

    # 6) Save replacement statistics to results/poisoned_data/{dataset}/poison_stats_{dataset}.json
    os.makedirs(poisoned_dir, exist_ok=True)
    stats_out_path = os.path.join(poisoned_dir, f"poison_stats_{dataset_name}.json")

    poison_stats_out = {
        "dataset": dataset_name,
        "overall": {
            "num_types": len(poison_stats),
            "num_entities": int(total_src),
            "total_freq": int(total_freq),
        },
        "per_type": poison_stats,
    }

    with open(stats_out_path, "w", encoding="utf-8") as f:
        json.dump(poison_stats_out, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Poison stats saved to {stats_out_path}")


def run_poison(
    dataset_names: List[str],
    data_root: str = "datasets",
    corpus_stats_root: str = "results/corpus_entities",
    queries_entities_root: str = "results/queries_entities",
    poisoned_root: str = "results/poisoned_data",
    top_ratio: float = 0.05,
):
    for dataset_name in dataset_names:
        logicPoison(
            dataset_name=dataset_name,
            data_root=data_root,
            corpus_stats_root=corpus_stats_root,
            queries_entities_root=queries_entities_root,
            poisoned_root=poisoned_root,
            top_ratio=top_ratio,
        )


def list_datasets(data_root: str = "datasets") -> List[str]:
    if not os.path.isdir(data_root):
        return []
    return sorted(
        d
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and os.path.isfile(os.path.join(data_root, d, "corpus.jsonl"))
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run logic-poison stage for selected datasets.")
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--corpus_stats_root", type=str, default="results/corpus_entities")
    parser.add_argument("--queries_entities_root", type=str, default="results/queries_entities")
    parser.add_argument("--poisoned_root", type=str, default="results/poisoned_data")
    parser.add_argument("--top_ratio", type=float, default=0.05)
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

    run_poison(
        dataset_names=selected,
        data_root=args.data_root,
        corpus_stats_root=args.corpus_stats_root,
        queries_entities_root=args.queries_entities_root,
        poisoned_root=args.poisoned_root,
        top_ratio=args.top_ratio,
    )


if __name__ == "__main__":
    main()

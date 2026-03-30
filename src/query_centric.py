import argparse
import ast
import glob
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional

import openai
from openai import OpenAI
from tqdm import tqdm

def _load_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return key


def _load_base_url() -> Optional[str]:
    url = os.getenv("OPENAI_BASE_URL", "").strip()
    return url or None


client = OpenAI(api_key=_load_api_key(), base_url=_load_base_url())


def make_prompt(q: str) -> str:
    tpl = """
Extract the reasoning-critical entities from the multi-hop question Q.

Your constraints:
- Do NOT answer Q.
- Do NOT add external knowledge.
- Use ONLY the wording of Q.
- Identify the minimal reasoning hops (1, 2, 3, ...).
- For each hop, extract only entities necessary for the reasoning chain.

Entity types:
Use spaCy's NER label set when applicable:
"PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
"EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL".

Additionally allow two reasoning-specific types:
- "ALIAS": paraphrased or descriptive label referring to an entity
- "BRIDGE": implicit descriptive entity needed to connect reasoning hops

Guidelines:
- Include implicit bridge entities (BRIDGE) and alias-style descriptions (ALIAS)
  if they are part of the reasoning chain.
- Do NOT normalize entities to real-world names.
- Focus only on entities whose presence or value matters for understanding or
  resolving the question.

For each extracted entity output:
- "hop": integer
- "entity": exact span or descriptive phrase from Q
- "type": one of the spaCy labels above OR "ALIAS" / "BRIDGE"

Output format (strict):
A Python-style list of dicts, e.g.:

[
  {"hop": 1, "entity": "...", "type": "GPE"},
  {"hop": 2, "entity": "...", "type": "BRIDGE"},
  {"hop": 3, "entity": "...", "type": "DATE"}
]

Return ONLY the list.

Q: {QUESTION_TEXT}
""".strip()
    return tpl.replace("{QUESTION_TEXT}", q)


def _parse_entities(txt: str):
    def _try_parse(s: str):
        for parser in (ast.literal_eval, json.loads):
            try:
                out = parser(s)
                if isinstance(out, list):
                    return out
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
        return None

    out = _try_parse(txt)
    if out is not None:
        return out

    try:
        i = txt.index("[")
        j = txt.rindex("]") + 1
        return _try_parse(txt[i:j])
    except ValueError:
        return None


def extract_entities(
    question: str,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    msg = make_prompt(question)
    rsp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": msg}],
        temperature=0.0,
    )
    raw = rsp.choices[0].message.content.strip()

    ents = _parse_entities(raw)
    if ents is None or not isinstance(ents, list):
        raise ValueError(f"Failed to parse entities; raw output:\n{raw}")

    rows = []
    for it in ents:
        if not isinstance(it, dict):
            continue
        hop = int(it.get("hop", 1))
        ent = str(it.get("entity", "")).strip()
        typ = str(it.get("type", "OTHER")).strip()
        if ent:
            rows.append({"hop": hop, "entity": ent, "type": typ})
    return rows


def extract_retry(
    item: Dict[str, Any],
    retries: int = 3,
    base_delay: float = 1.5,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    qid = item.get("_id")
    qtxt = item.get("text", "")

    for i in range(1, retries + 1):
        try:
            ents = extract_entities(qtxt, model=model)
            return {"_id": qid, "text": qtxt, "entities": ents, "error": None}
        except (ValueError, openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as err:
            if i == retries:
                return {"_id": qid, "text": qtxt, "entities": None, "error": str(err)}
            wait = base_delay * (2 ** (i - 1))
            time.sleep(wait + random.uniform(0, 0.2))


def run_query_file(
    input_path: str,
    output_path: str,
    max_workers: int = 8,
    queue_factor: int = 4,
    model: str = "gpt-4o-mini",
):
    log = logging.getLogger("entity_extraction")
    if not log.handlers:
        log.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
        h.setFormatter(fmt)
        log.addHandler(h)

    log.info(f"Counting lines in {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        n_q = sum(1 for _ in f)
    log.info(f"Found {n_q} questions in {input_path}")

    max_jobs = max_workers * queue_factor

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=max_workers) as executor, \
         tqdm(total=n_q, desc="Processing questions", unit="q") as pbar:

        jobs = {}

        def handle_finished_future(fut):
            res = fut.result()
            if res.get("error"):
                log.warning(f"QID={res.get('_id')} extraction error: {res['error']}")
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()
            pbar.update(1)

        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec: Dict[str, Any] = json.loads(ln)
            except json.JSONDecodeError as err:
                log.error(f"Failed to parse line as JSON: {err} | line={ln[:100]}")
                continue

            job = executor.submit(
                extract_retry,
                rec,
                3,
                1.5,
                model,
            )
            jobs[job] = rec.get("_id")

            if len(jobs) >= max_jobs:
                for done in as_completed(list(jobs.keys())):
                    handle_finished_future(done)
                    del jobs[done]
                    break

        log.info("All tasks submitted, waiting for remaining results ...")
        for done in as_completed(list(jobs.keys())):
            handle_finished_future(done)

        log.info(f"All done. Results saved to {output_path}")


def run_queries(
    dataset_names: Iterable[str],
    data_root: str = "datasets",
    output_dir: str = "results/queries_entities",
    max_workers: int = 8,
    queue_factor: int = 4,
    model: str = "gpt-4o-mini",
):
    os.makedirs(output_dir, exist_ok=True)

    for ds in dataset_names:
        in_path = os.path.join(data_root, ds, "queries.jsonl")
        out_path = os.path.join(output_dir, f"{ds}.jsonl")

        print(f"[INFO] Processing dataset {ds}")
        print(f"       Input : {in_path}")
        print(f"       Output: {out_path}")

        run_query_file(
            input_path=in_path,
            output_path=out_path,
            max_workers=max_workers,
            queue_factor=queue_factor,
            model=model,
        )

    print("[INFO] Query stage finished.")


def list_datasets(data_root: str) -> List[str]:
    pat = os.path.join(data_root, "*/queries.jsonl")
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(pat))


def parse_args():
    parser = argparse.ArgumentParser(description="Batch extract entities from queries.jsonl files.")
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--queue_factor", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/queries_entities")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
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

    run_queries(
        selected,
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        queue_factor=args.queue_factor,
        model=args.model,
    )


if __name__ == "__main__":
    main()

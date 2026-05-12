import argparse
import ast
import glob
import json
import os
import random
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Any

import yaml
from openai import OpenAI
from tqdm import tqdm


def _load_graphrag_settings(settings_path: str = None) -> Dict[str, Any]:
    """Load graphrag settings.yaml to get LLM configuration."""
    if settings_path is None:
        # Try to find settings.yaml in graphrag-api
        possible_paths = [
            "graphrag-api/graphrag/settings.yaml",
            "../graphrag-api/graphrag/settings.yaml",
            "../../graphrag-api/graphrag/settings.yaml",
        ]
        for path in possible_paths:
            if os.path.isfile(path):
                settings_path = path
                break
        if settings_path is None:
            raise FileNotFoundError("Could not find graphrag settings.yaml")
    
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)
    return settings


def _get_llm_client(settings: Dict[str, Any]) -> tuple:
    """Extract LLM client config from graphrag settings."""
    model_config = settings.get("models", {}).get("default_chat_model", {})
    
    api_base = model_config.get("api_base")
    api_key = model_config.get("api_key", "").strip()
    model_name = model_config.get("model")
    
    # Resolve ${GRAPHRAG_API_KEY} placeholder
    if api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.getenv(env_var, "").strip()
        if not api_key:
            raise RuntimeError(f"Environment variable {env_var} is not set.")
    
    client = OpenAI(api_key=api_key, base_url=api_base)
    return client, model_name


def _extract_entities_llm(text: str, client: OpenAI, model: str) -> Dict[str, List[str]]:
    """Extract entities from text using LLM. Returns {label: [entities]}"""
    prompt = f"""Extract named entities from the following text. For each entity, identify its type (PERSON, ORG, GPE, LOCATION, PRODUCT, EVENT, etc.).

Return a JSON object with entity types as keys and lists of entities as values. Only include entities that actually appear in the text.

Text: {text}

Return ONLY valid JSON, e.g.:
{{"PERSON": ["John", "Mary"], "ORG": ["Google"], "GPE": ["USA"]}}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            entities = json.loads(result_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            try:
                start = result_text.index("{")
                end = result_text.rindex("}") + 1
                entities = json.loads(result_text[start:end])
            except (ValueError, json.JSONDecodeError):
                return {}
        
        # Ensure all values are lists
        return {k: (v if isinstance(v, list) else [v]) for k, v in entities.items() if isinstance(v, (list, str))}
    except Exception as e:
        print(f"[WARN] LLM extraction failed: {e}")
        return {}


def corpus_stats(
    input_path: str,
    batch_size: int = 32,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Dict[str, Counter]:
    """
    Process corpus.jsonl using LLM-based NER.
    Returns:
        { label: Counter({ entity_text: count }) }
    """
    if client is None or model is None:
        # Load settings if not provided
        settings = _load_graphrag_settings()
        client, model = _get_llm_client(settings)
    
    cnts: Dict[str, Counter] = defaultdict(Counter)

    with open(input_path, "r", encoding="utf-8") as f:
        n_docs = sum(1 for _ in f)
    print(f"[INFO] corpus size: {n_docs} docs")

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=n_docs, desc=f"NER (LLM) on {os.path.basename(input_path)}", unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                txt = rec.get("text", "")
                if not txt:
                    continue
                
                # Extract entities using LLM
                entities = _extract_entities_llm(txt, client, model)
                for label, entity_list in entities.items():
                    for entity in entity_list:
                        if isinstance(entity, str):
                            cnts[label][entity.strip()] += 1
            except Exception as e:
                print(f"[WARN] Error processing document: {e}")
                continue

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
    
    # Load graphrag settings and LLM client once
    settings = _load_graphrag_settings()
    client, model = _get_llm_client(settings)
    print(f"[INFO] Using LLM model: {model}")

    for ds in dataset_names:
        in_path = os.path.join(data_root, ds, "corpus.jsonl")
        out_path = os.path.join(output_root, f"{ds}.json")

        print(f"\n[INFO] Processing corpus for dataset: {ds}")
        print(f"       Input : {in_path}")
        print(f"       Output: {out_path}")

        cnts = corpus_stats(
            in_path,
            batch_size=batch_size,
            client=client,
            model=model,
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

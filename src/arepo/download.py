#!/usr/bin/env python3
"""Download benchmark datasets and pre-download model weights."""

import os
import sys
import json
import subprocess
import urllib.request
from pathlib import Path


def download_cgtd(target_dir):
    """Download the CGTD benchmark dataset."""
    target = Path(target_dir) / "cgtd"
    if target.exists() and any(target.iterdir()):
        print(f"CGTD already exists at {target}")
        return True

    print("Downloading CGTD benchmark...")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone",
             "https://github.com/rexshijaku/chatgpt-generated-text-detection-corpus.git",
             str(target)],
            check=True
        )
        # Verify
        human_dir = target / "full_texts" / "human"
        chatgpt_dir = target / "full_texts" / "chatgpt"
        if human_dir.exists() and chatgpt_dir.exists():
            n_human = len(list(human_dir.glob("*.txt")))
            n_ai = len(list(chatgpt_dir.glob("*.txt")))
            print(f"CGTD downloaded: {n_human} human, {n_ai} AI texts")
            return True
        else:
            print("ERROR: CGTD downloaded but expected directories not found")
            return False
    except FileNotFoundError:
        print("ERROR: git not found. Install git and try again.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git clone failed: {e}")
        return False


def download_hc3(target_dir):
    """Download HC3 (Human-ChatGPT Comparison Corpus) dataset."""
    target = Path(target_dir) / "hc3"
    data_file = target / "all.jsonl"
    if data_file.exists() and data_file.stat().st_size > 0:
        print(f"HC3 already exists at {data_file}")
        return True

    print("Downloading HC3 dataset...")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    try:
        url = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
        urllib.request.urlretrieve(url, data_file)
        print(f"HC3 downloaded to {data_file}")
        return True
    except Exception as e:
        print(f"ERROR: HC3 download failed: {e}")
        return False


def download_wikipedia(target_dir):
    """Download Wikipedia AI-generated text dataset."""
    target = Path(target_dir) / "wikipedia_ai"
    marker = target / ".complete"
    if marker.exists():
        print(f"Wikipedia dataset already exists at {target}")
        return True

    print("Downloading GPT-Wiki-Intro dataset...")
    target.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "aadityaubhat/GPT-wiki-intro",
            split="train",
            cache_dir=str(target),
        )
        # Force materialization and verify expected schema.
        first = dataset[0]
        if "wiki_intro" not in first or "generated_intro" not in first:
            print("ERROR: GPT-Wiki-Intro downloaded but expected columns not found")
            return False
        marker.write_text("ok\n", encoding="utf-8")
        print(f"GPT-Wiki-Intro cached at {target}")
        return True
    except Exception as e:
        print(f"ERROR: GPT-Wiki-Intro download failed: {e}")
        return False


def load_sample_texts():
    """Load bundled sample texts.

    Returns
    -------
    human_texts : list of str
    ai_texts : list of str
    """
    data_dir = get_data_dir()
    human_texts = []
    for f in sorted(data_dir.glob("nat_text_*.txt")):
        human_texts.append(f.read_text(encoding="utf-8").strip())
    ai_texts = []
    for f in sorted(data_dir.glob("gpt_text_*.txt")):
        ai_texts.append(f.read_text(encoding="utf-8").strip())
    return human_texts, ai_texts


def load_cgtd(benchmarks_dir="benchmarks", max_per_class=None, auto_download=True):
    """Load CGTD benchmark texts, downloading if missing.

    Parameters
    ----------
    benchmarks_dir : str
        Parent directory for benchmark datasets.
    max_per_class : int or None
        Cap per class. None means load all.
    auto_download : bool
        If True, download CGTD when missing.

    Returns
    -------
    human_texts : list of str
    ai_texts : list of str
    """
    cgtd_dir = Path(benchmarks_dir) / "cgtd"
    human_dir = cgtd_dir / "full_texts" / "human"
    chatgpt_dir = cgtd_dir / "full_texts" / "chatgpt"

    if not human_dir.exists() or not chatgpt_dir.exists():
        if auto_download:
            ok = download_cgtd(benchmarks_dir)
            if not ok:
                raise RuntimeError("Failed to download CGTD benchmark")
        else:
            raise FileNotFoundError(
                f"CGTD not found at {cgtd_dir}. Run arepo-download or set auto_download=True"
            )

    human_texts = []
    for f in sorted(human_dir.glob("*.txt")):
        text = f.read_text(encoding="utf-8").strip()
        if text:
            human_texts.append(text)
        if max_per_class and len(human_texts) >= max_per_class:
            break

    ai_texts = []
    for f in sorted(chatgpt_dir.glob("*.txt")):
        text = f.read_text(encoding="utf-8").strip()
        if text:
            ai_texts.append(text)
        if max_per_class and len(ai_texts) >= max_per_class:
            break

    return human_texts, ai_texts


def load_hc3(max_per_class=50, benchmarks_dir="benchmarks", auto_download=True, skip_per_class=0):
    """Load HC3 dataset from a local JSONL cache.

    Parameters
    ----------
    max_per_class : int
        Maximum texts per class.
    benchmarks_dir : str
        Parent directory for benchmark datasets.
    auto_download : bool
        If True, download HC3 when missing.

    Returns
    -------
    human_texts : list of str
    ai_texts : list of str
    """
    data_file = Path(benchmarks_dir) / "hc3" / "all.jsonl"
    if not data_file.exists():
        if auto_download:
            ok = download_hc3(benchmarks_dir)
            if not ok:
                raise RuntimeError("Failed to download HC3 benchmark")
        else:
            raise FileNotFoundError(
                f"HC3 not found at {data_file}. Run arepo-download or set auto_download=True"
            )

    human_texts = []
    ai_texts = []
    skipped_human = 0
    skipped_ai = 0

    with data_file.open("r", encoding="utf-8") as fh:
        examples = (json.loads(line) for line in fh if line.strip())
        for example in examples:
            if len(human_texts) >= max_per_class and len(ai_texts) >= max_per_class:
                break

            question = example.get("question", "")

            if "human_answers" in example and example["human_answers"]:
                for answer in example["human_answers"]:
                    if len(human_texts) >= max_per_class:
                        break
                    text = f"{question}\n\n{answer}" if question else answer
                    if text and len(text) > 100:
                        if skipped_human < skip_per_class:
                            skipped_human += 1
                        else:
                            human_texts.append(text)

            if "chatgpt_answers" in example and example["chatgpt_answers"]:
                for answer in example["chatgpt_answers"]:
                    if len(ai_texts) >= max_per_class:
                        break
                    text = f"{question}\n\n{answer}" if question else answer
                    if text and len(text) > 100:
                        if skipped_ai < skip_per_class:
                            skipped_ai += 1
                        else:
                            ai_texts.append(text)

    return human_texts, ai_texts


def load_wiki(max_per_class=50, benchmarks_dir="benchmarks", skip_per_class=0):
    """Load GPT-Wiki-Intro dataset from HuggingFace.

    Parameters
    ----------
    max_per_class : int
        Maximum texts per class.
    benchmarks_dir : str
        Parent directory for benchmark dataset caches.

    Returns
    -------
    human_texts : list of str
    ai_texts : list of str
    """
    from datasets import load_dataset

    cache_dir = Path(benchmarks_dir) / "wikipedia_ai"
    dataset = load_dataset(
        "aadityaubhat/GPT-wiki-intro",
        split="train",
        cache_dir=str(cache_dir),
    )

    human_texts = []
    ai_texts = []
    skipped_human = 0
    skipped_ai = 0

    for example in dataset:
        if len(human_texts) >= max_per_class and len(ai_texts) >= max_per_class:
            break

        if len(human_texts) < max_per_class:
            text = example.get("wiki_intro", "")
            if text and len(text) > 100:
                if skipped_human < skip_per_class:
                    skipped_human += 1
                else:
                    human_texts.append(text)

        if len(ai_texts) < max_per_class:
            text = example.get("generated_intro", "")
            if text and len(text) > 100:
                if skipped_ai < skip_per_class:
                    skipped_ai += 1
                else:
                    ai_texts.append(text)

    return human_texts, ai_texts


def load_raid(max_per_class=50, split="train", skip_per_class=0):
    """Load RAID human/AI generations from HuggingFace.

    RAID marks human rows with ``model == "human"``; all other model values are
    generated text. Text is stored in the ``generation`` field.
    """
    from datasets import load_dataset

    dataset = load_dataset("liamdugan/raid", split=split, streaming=True)
    human_texts = []
    ai_texts = []
    skipped_human = 0
    skipped_ai = 0

    for example in dataset:
        if len(human_texts) >= max_per_class and len(ai_texts) >= max_per_class:
            break
        text = (example.get("generation") or "").strip()
        if len(text) <= 100:
            continue
        if example.get("model") == "human":
            if len(human_texts) < max_per_class:
                if skipped_human < skip_per_class:
                    skipped_human += 1
                else:
                    human_texts.append(text)
        elif len(ai_texts) < max_per_class:
            if skipped_ai < skip_per_class:
                skipped_ai += 1
            else:
                ai_texts.append(text)

    return human_texts, ai_texts


def load_mage(max_per_class=50, split="train", skip_per_class=0):
    """Load MAGE benchmark text from HuggingFace.

    MAGE uses ``label == 1`` for human text and ``label == 0`` for machine text.
    """
    from datasets import load_dataset

    dataset = load_dataset("yaful/MAGE", split=split, streaming=True)
    human_texts = []
    ai_texts = []
    skipped_human = 0
    skipped_ai = 0

    for example in dataset:
        if len(human_texts) >= max_per_class and len(ai_texts) >= max_per_class:
            break
        text = (example.get("text") or "").strip()
        if len(text) <= 100:
            continue
        if example.get("label") == 1:
            if len(human_texts) < max_per_class:
                if skipped_human < skip_per_class:
                    skipped_human += 1
                else:
                    human_texts.append(text)
        elif example.get("label") == 0 and len(ai_texts) < max_per_class:
            if skipped_ai < skip_per_class:
                skipped_ai += 1
            else:
                ai_texts.append(text)

    return human_texts, ai_texts


def load_arepo_essays(max_per_class=50, split="train", skip_per_class=0):
    """Load Arepo essay benchmark text from HuggingFace.

    In this cleaned release, ``label == 0`` is human and all nonzero labels are
    generated variants from different models/prompts.
    """
    from datasets import load_dataset

    dataset = load_dataset("polsci/ghostbuster-essay-cleaned", split=split, streaming=True)
    human_texts = []
    ai_texts = []
    skipped_human = 0
    skipped_ai = 0

    for example in dataset:
        if len(human_texts) >= max_per_class and len(ai_texts) >= max_per_class:
            break
        text = (example.get("text") or "").strip()
        if len(text) <= 100:
            continue
        if example.get("label") == 0:
            if len(human_texts) < max_per_class:
                if skipped_human < skip_per_class:
                    skipped_human += 1
                else:
                    human_texts.append(text)
        elif len(ai_texts) < max_per_class:
            if skipped_ai < skip_per_class:
                skipped_ai += 1
            else:
                ai_texts.append(text)

    return human_texts, ai_texts


def download_models():
    """Pre-download transformer model weights."""
    from transformers import AutoTokenizer, AutoModel
    from .core import DEFAULT_MODELS

    for model_name in DEFAULT_MODELS:
        print(f"Downloading {model_name}...")
        try:
            AutoTokenizer.from_pretrained(model_name)
            AutoModel.from_pretrained(model_name)
            print(f"  OK")
        except Exception as e:
            print(f"  FAILED: {e}")


def get_data_dir():
    """Return path to included sample data."""
    return Path(__file__).parent / "data"


def main():
    """CLI entrypoint: download benchmarks and models."""
    import argparse
    parser = argparse.ArgumentParser(description="Download Arepo datasets and models")
    parser.add_argument("--benchmarks-dir", default="benchmarks",
                        help="Directory for benchmark datasets (default: ./benchmarks)")
    parser.add_argument("--models", action="store_true",
                        help="Also pre-download transformer model weights")
    parser.add_argument("--all", action="store_true",
                        help="Download everything (benchmarks + models)")
    args = parser.parse_args()

    ok = True

    print("="*60)
    print("Arepo - Data Setup")
    print("="*60)

    # Sample data location
    data_dir = get_data_dir()
    if data_dir.exists():
        n_files = len(list(data_dir.glob("*.txt")))
        print(f"\nSample data: {n_files} files at {data_dir}")
    else:
        print(f"\nWARNING: Sample data not found at {data_dir}")

    # Benchmarks
    print()
    if not download_cgtd(args.benchmarks_dir):
        ok = False

    print()
    if not download_hc3(args.benchmarks_dir):
        ok = False

    print()
    if not download_wikipedia(args.benchmarks_dir):
        ok = False

    # Models
    if args.models or args.all:
        print()
        download_models()

    print()
    if ok:
        print("Setup complete!")
    else:
        print("Setup completed with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch the benchmark files that are not distributed through the HF Hub.

Downloaded into ``data/raw/`` (git-ignored -- we do not redistribute them):

* CrowS-Pairs  -- ``nyu-mll/crows-pairs`` on GitHub
* BBQ          -- ``nyu-mll/BBQ`` on GitHub, all nine categories
* COMPAS       -- ``propublica/compas-analysis`` on GitHub

StereoSet, BOLD, GSM8K, StrategyQA, ARC-easy, PIQA, Wikitext-2 and the TL;DR
summaries subset all come from the Hub and need no manual step.

The Utrecht fairness recruitment dataset is Kaggle-hosted, so it is fetched
through ``kagglehub`` (no credentials are needed for this public dataset) rather
than redistributed.  Pass ``--skip-utrecht`` to leave it out.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

GITHUB_FILES = {
    "crows_pairs.csv": "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv",
    "compas-scores-two-years.csv": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
}

BBQ_CATEGORIES = (
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
)
BBQ_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/{cat}.jsonl"

KAGGLE_DATASET = "ictinstitute/utrecht-fairness-recruitment-dataset"


def _download(url: str, dest: Path, force: bool = False) -> bool:
    if dest.exists() and not force and dest.stat().st_size > 0:
        print(f"  [skip] {dest.relative_to(ROOT)} already present")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as fh:
            fh.write(r.read())
    except Exception as exc:
        print(f"  [FAIL] {url}: {exc}")
        return False
    print(f"  [ok]   {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)")
    return True


def fetch_kaggle(force: bool = False) -> bool:
    """Fetch the Utrecht recruitment CSV.

    ``kagglehub`` serves this public dataset without credentials; the ``kaggle``
    CLI (which does need ``~/.kaggle/kaggle.json``) is the fallback.
    """
    out = RAW / "utrecht"
    if not force and out.exists() and any(out.glob("*.csv")):
        print(f"  [skip] {out.relative_to(ROOT)} already present")
        return True
    out.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub

        src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        n = 0
        for csv in src.glob("*.csv"):
            shutil.copy2(csv, out / csv.name)
            n += 1
        if n:
            print(f"  [ok]   {out.relative_to(ROOT)} ({n} csv via kagglehub)")
            return True
    except Exception as exc:
        print(f"  [warn] kagglehub failed ({str(exc)[:90]}); trying the kaggle CLI")

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(out), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        print("  [FAIL] neither kagglehub nor the kaggle CLI is available "
              "(pip install kagglehub)")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"  [FAIL] kaggle download failed ({exc}); check ~/.kaggle/kaggle.json")
        return False
    ok = any(out.glob("*.csv"))
    print(f"  [{'ok' if ok else 'FAIL'}]   {out.relative_to(ROOT)}")
    return ok


def prefetch_hub() -> None:
    """Warm the Hub-hosted datasets so later runs are offline-friendly."""
    from datasets import load_dataset

    targets = [
        ("McGill-NLP/stereoset", "intersentence", "validation"),
        ("AlexaAI/bold", None, "train"),
        ("openai/gsm8k", "main", "test"),
        ("ChilleD/StrategyQA", None, "train"),
        ("allenai/ai2_arc", "ARC-Easy", "test"),
        ("baber/piqa", None, "validation"),
        ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
        ("CarperAI/openai_summarize_tldr", None, "test"),
    ]
    for repo, cfg, split in targets:
        try:
            ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
            print(f"  [ok]   {repo} ({len(ds)} rows)")
        except Exception as exc:
            print(f"  [FAIL] {repo}: {str(exc)[:120]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--skip-utrecht", action="store_true", help="do not fetch the Utrecht dataset")
    ap.add_argument("--skip-hub", action="store_true", help="do not prefetch Hub datasets")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    print("GitHub-hosted benchmarks:")
    ok = True
    for name, url in GITHUB_FILES.items():
        ok &= _download(url, RAW / name, args.force)

    print("BBQ categories:")
    for cat in BBQ_CATEGORIES:
        ok &= _download(BBQ_URL.format(cat=cat), RAW / "bbq" / f"{cat}.jsonl", args.force)

    if not args.skip_utrecht:
        print("Utrecht (Kaggle):")
        ok &= fetch_kaggle(args.force)
    else:
        print("Utrecht: skipped by request.")

    if not args.skip_hub:
        print("Hub-hosted datasets:")
        prefetch_hub()

    print("\ndone." if ok else "\ndone, with failures above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

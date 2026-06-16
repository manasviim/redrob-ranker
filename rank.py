#!/usr/bin/env python3
"""
rank.py — Phase 2 ranking: reads candidates.jsonl, outputs submission.csv
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from features import extract_features
from parse_jd import get_jd_profile, get_jd_text

# ── Config ────────────────────────────────────────────────────────────────────

TOP_N_FOR_TFIDF = 500   # apply TF-IDF re-rank on top-N by primary score
FINAL_SHORTLIST = 100   # output top-100
WEIGHTS = get_jd_profile()["weights"]

# ── Loading ───────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> List[Dict[str, Any]]:
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def build_tfidf_scores(candidates_subset: List[Dict[str, Any]]) -> np.ndarray:
    """Fit TF-IDF on subset, return cosine similarity vs JD query."""
    jd_text = get_jd_text()
    texts = [c["text"] for c in candidates_subset]
    all_texts = [jd_text] + texts

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=30000,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    jd_vec = tfidf_matrix[0]
    cand_vecs = tfidf_matrix[1:]
    sims = cosine_similarity(jd_vec, cand_vecs).flatten()
    # Normalize to 0-1
    if sims.max() > 0:
        sims = sims / sims.max()
    return sims


# ── Weighted scoring ──────────────────────────────────────────────────────────

def weighted_score(feat: Dict[str, Any], tfidf_s: float) -> float:
    return (
        feat["skill_score"] * WEIGHTS["skill"]
        + feat["career_score"] * WEIGHTS["career"]
        + tfidf_s * WEIGHTS["tfidf"]
        + feat["behavior_score"] * WEIGHTS["behavior"]
    )


# ── Reasoning text ────────────────────────────────────────────────────────────

def make_reasoning(feat: Dict[str, Any], final_score: float) -> str:
    title = feat["current_title"] or "Unknown title"
    yoe = feat["yoe"]
    loc = feat["location"] or "Unknown"
    rr = feat["response_rate"]
    otw = "open-to-work" if feat["open_to_work"] else "not open-to-work"
    notice = feat["notice_days"]

    return (
        f"{title} with {yoe:.1f} yrs; "
        f"skill={feat['skill_score']:.2f}, career={feat['career_score']:.2f}, "
        f"behavior={feat['behavior_score']:.2f}; "
        f"location={loc}; response_rate={rr:.2f}; {otw}; notice={notice}d."
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_ranking(candidates_path: str, out_path: str, verbose: bool = True):
    t0 = time.time()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    if verbose:
        print(f"[1/5] Loading candidates from {candidates_path} …", flush=True)
    candidates = load_candidates(candidates_path)
    if verbose:
        print(f"      Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # ── 2. Extract features ──────────────────────────────────────────────────
    if verbose:
        print("[2/5] Extracting features …", flush=True)
    t1 = time.time()
    features = []
    for i, c in enumerate(candidates):
        feat = extract_features(c)
        features.append(feat)
        if verbose and (i + 1) % 10000 == 0:
            print(f"      … {i+1:,} done ({time.time()-t1:.1f}s)", flush=True)

    if verbose:
        print(f"      Done in {time.time()-t1:.1f}s")

    # ── 3. Primary weighted score (skill + career + behavior; tfidf=0 for now) ──
    if verbose:
        print("[3/5] Primary scoring …", flush=True)
    for feat in features:
        feat["primary_score"] = weighted_score(feat, tfidf_s=0.0)

    # ── 4. Honeypot filter + top-500 for TF-IDF ──────────────────────────────
    if verbose:
        print("[4/5] Honeypot filter + TF-IDF re-rank on top-500 …", flush=True)

    clean = [f for f in features if not f["is_honeypot"]]
    if verbose:
        honeypot_count = len(features) - len(clean)
        print(f"      Removed {honeypot_count} honeypot profiles")

    # Sort by primary score, take top 500
    clean.sort(key=lambda x: x["primary_score"], reverse=True)
    top500 = clean[:TOP_N_FOR_TFIDF]

    # Build TF-IDF scores for top-500
    tfidf_scores = build_tfidf_scores(top500)

    # Final score with TF-IDF
    for i, feat in enumerate(top500):
        feat["final_score"] = weighted_score(feat, tfidf_s=float(tfidf_scores[i]))

    # ── 5. Final sort & output ────────────────────────────────────────────────
    if verbose:
        print("[5/5] Producing submission.csv …", flush=True)

    top500.sort(key=lambda x: (-x["final_score"], x["candidate_id"].upper()))
    final_100 = top500[:FINAL_SHORTLIST]

    # Normalize scores to 0-1 range, highest = ~0.99
    max_score = final_100[0]["final_score"] if final_100 else 1.0
    min_score = final_100[-1]["final_score"] if final_100 else 0.0
    score_range = max_score - min_score if max_score != min_score else 1.0

    rows = []
    for rank, feat in enumerate(final_100, start=1):
        # Scale to 0.50–0.99 range; subtract tiny offset to guarantee strictly decreasing scores
        normalized = 0.50 + 0.49 * (feat["final_score"] - min_score) / score_range
        normalized = round(normalized - (rank - 1) * 0.000001, 6)
        rows.append({
            "candidate_id": feat["candidate_id"],
            "rank": rank,
            "score": normalized,
            "reasoning": make_reasoning(feat, normalized),
        })

    out_file = Path(out_path)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n✅  Done in {elapsed:.1f}s — {out_path}")
        print(f"    Top-3 candidates:")
        for r in rows[:3]:
            print(f"      #{r['rank']} {r['candidate_id']} score={r['score']} — {r['reasoning'][:80]}…")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv",
                        help="Output CSV path")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    run_ranking(args.candidates, args.out, verbose=not args.quiet)

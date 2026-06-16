# Redrob Hackathon — Intelligent Candidate Ranker

A multi-signal structured ranker that evaluates candidates the way a senior recruiter would — not by keyword matching, but by reasoning about skill depth, career quality, domain fit, and behavioral availability.

## Approach

### Problem with keyword matching
Keyword-only rankers rank an HR Manager who lists "embeddings" in their skills above a real ML Engineer who built RAG systems but described it differently. Our system avoids this.

### Our signals
Three scoring components, each grounded in the JD:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Skill score** | 38% | Must-have skill coverage (depth × proficiency × duration), nice-to-have bonus, anti-domain penalty (CV/speech/robotics), Redrob assessment scores |
| **Career score** | 32% | YoE in 5–9 ideal range, product company vs consulting history, title domain match, GitHub/open-source activity |
| **TF-IDF similarity** | 20% | Semantic similarity between full candidate text and a JD-derived query (bigrams, sublinear TF) |
| **Behavior score** | 10% | Last active recency, open-to-work flag, response rate, notice period, location fit |

Plus a **honeypot multiplier** that penalizes impossible profiles (expert skills with 0 months use, career months inconsistent with YoE, etc.).

### What we explicitly do NOT do
- No LLM API calls
- No GPU
- No network during ranking
- No keyword-count tricks — proficiency and duration are weighted

## Reproduce

### Requirements
```
scikit-learn>=1.3
```

Install:
```bash
pip install -r requirements.txt
```

### Run
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- Input: `candidates.jsonl` (100K lines, one JSON object per line)
- Output: `submission.csv` (100 rows: candidate_id, rank, score, reasoning)
- Runtime: ~14 seconds on CPU
- Memory: < 2 GB RAM

### Validate
```bash
python validate_submission.py submission.csv
```

## Output format
```
candidate_id,rank,score,reasoning
CAND_0012345,1,0.6831,"Senior ML Engineer at Zomato, 7.2 yrs (ideal range). Matches JD on: weaviate, pinecone, learning to rank; notice ≤30d; strong GitHub activity (94.8)."
...
```

## Architecture decisions

**Why TF-IDF instead of dense embeddings?**
Dense embedding models (sentence-transformers etc.) take 3–10 minutes to encode 100K candidates on CPU. TF-IDF with bigrams over a pre-scored pool of 3000 captures the same semantic signal in ~2 seconds and is fully explainable.

**Why a structured scorer before TF-IDF?**
Running TF-IDF over all 100K would add noise from irrelevant candidates. We score all 100K with structured signals first, then apply TF-IDF as a re-ranker on the top 3000 — giving both speed and quality.

**Why explicit anti-signals?**
The JD explicitly names wrong-domain candidates (CV, speech, robotics) and consulting-only backgrounds. Encoding these as hard penalties rather than just "less skill match" is more faithful to what a recruiter would actually do.

## Files
```
rank.py                  — main ranker (single command, no dependencies except scikit-learn)
requirements.txt         — pinned dependencies
submission.csv           — our submission
submission_metadata.yaml — portal metadata
README.md                — this file
```

"""
features.py — Extract all numeric features from a candidate record.
"""

from __future__ import annotations
import math
from datetime import date, datetime
from typing import Any, Dict

from parse_jd import get_jd_profile

JD = get_jd_profile()

# ── helpers ──────────────────────────────────────────────────────────────────

def _today() -> date:
    return date.today()


def _days_since(date_str: str) -> float:
    """Days since a date string (YYYY-MM-DD). Returns large number on error."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (_today() - d).days
    except Exception:
        return 9999


def _lower_set(lst: list) -> set:
    return {s.lower() for s in lst if isinstance(s, str)}


def _build_candidate_text(c: Dict[str, Any]) -> str:
    """Single text blob for TF-IDF."""
    parts = []
    p = c.get("profile", {})
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(p.get("current_title", ""))
    parts.append(p.get("current_industry", ""))
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("industry", ""))
    for sk in c.get("skills", []):
        parts.append(sk.get("name", ""))
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))
    return " ".join(parts)


# ── Skill match score (0-1) ──────────────────────────────────────────────────

def skill_score(c: Dict[str, Any]) -> float:
    skills_raw = c.get("skills", [])
    proficiency_weight = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.9, "expert": 1.0}

    # Build weighted skill presence dict
    skill_strengths: Dict[str, float] = {}
    for sk in skills_raw:
        name = sk.get("name", "").lower()
        prof = proficiency_weight.get(sk.get("proficiency", "beginner"), 0.3)
        endorsements = min(sk.get("endorsements", 0), 100)
        duration = min(sk.get("duration_months", 0), 60)
        # Trust multiplier: endorsements + duration boost
        trust = 1.0 + (endorsements / 200) + (duration / 120)
        skill_strengths[name] = min(prof * trust, 1.5)  # cap

    # Must-have skills — each contributes equally
    must_haves = JD["must_have_skills"]
    must_score = 0.0
    for req in must_haves:
        req_l = req.lower()
        # Exact match or substring in skill name
        match_val = 0.0
        for sname, strength in skill_strengths.items():
            if req_l in sname or sname in req_l:
                match_val = max(match_val, strength)
        must_score += min(match_val, 1.0)
    must_score = (must_score / len(must_haves)) if must_haves else 0.0

    # Nice-to-have skills (capped bonus)
    nice_haves = JD["nice_to_have_skills"]
    nice_score = 0.0
    for req in nice_haves:
        req_l = req.lower()
        for sname, strength in skill_strengths.items():
            if req_l in sname or sname in req_l:
                nice_score += 0.5
                break
    nice_score = min(nice_score / len(nice_haves), 1.0) if nice_haves else 0.0

    # Anti-skills penalty
    anti_skills = JD["anti_skills"]
    anti_penalty = 0.0
    for anti in anti_skills:
        anti_l = anti.lower()
        for sname, strength in skill_strengths.items():
            if anti_l in sname or sname in anti_l:
                anti_penalty += strength * 0.15
    anti_penalty = min(anti_penalty, 0.4)

    # Also check career history text for must-have signals
    career_text = " ".join(
        (j.get("description", "") + " " + j.get("title", "")).lower()
        for j in c.get("career_history", [])
    )
    summary_text = c.get("profile", {}).get("summary", "").lower()
    context_text = career_text + " " + summary_text

    context_boost = 0.0
    for req in must_haves:
        if req.lower() in context_text:
            context_boost += 0.01  # small boost per mention
    context_boost = min(context_boost, 0.15)

    raw = must_score * 0.7 + nice_score * 0.3 + context_boost - anti_penalty
    return max(0.0, min(raw, 1.0))


# ── Career quality score (0-1) ───────────────────────────────────────────────

def career_score(c: Dict[str, Any]) -> float:
    history = c.get("career_history", [])
    profile = c.get("profile", {})
    yoe = profile.get("years_of_experience", 0)

    # 1. Years-of-experience fit
    yoe_min, yoe_max = JD["yoe_min"], JD["yoe_max"]
    ideal_min, ideal_max = JD["yoe_ideal_min"], JD["yoe_ideal_max"]
    if ideal_min <= yoe <= ideal_max:
        yoe_s = 1.0
    elif yoe_min <= yoe <= yoe_max:
        yoe_s = 0.8
    elif yoe < yoe_min:
        yoe_s = max(0.0, 0.5 * (yoe / yoe_min))
    else:  # > yoe_max
        yoe_s = max(0.4, 1.0 - 0.05 * (yoe - yoe_max))

    # 2. Product company vs consulting
    anti_cos = {a.lower() for a in JD["anti_company_types"]}
    product_industries = {p.lower() for p in JD["product_company_industries"]}

    consulting_flag = False
    product_months = 0
    total_months = 0
    for job in history:
        co = job.get("company", "").lower()
        ind = job.get("industry", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur
        # Check if pure consulting
        if any(a in co for a in anti_cos):
            consulting_flag = True
        else:
            # Check if product company
            if any(pi in ind for pi in product_industries):
                product_months += dur
            else:
                # Unclear — give partial credit
                product_months += dur * 0.4

    product_ratio = (product_months / total_months) if total_months > 0 else 0.0
    product_s = product_ratio

    if consulting_flag and product_months < 24:
        # Mostly consulting career — hard penalty
        product_s *= 0.3

    # 3. Title trajectory — does it show ML/AI growth?
    ai_title_keywords = [
        "ml", "machine learning", "ai", "data scientist", "nlp",
        "research", "engineer", "ranking", "search", "recommendation",
        "backend", "software", "applied"
    ]
    relevant_titles = 0
    for job in history:
        title = job.get("title", "").lower()
        if any(kw in title for kw in ai_title_keywords):
            relevant_titles += 1
    title_s = min(relevant_titles / max(len(history), 1), 1.0)

    # 4. Stability — penalize very short stints (< 12 months) on multiple jobs
    short_stints = sum(1 for j in history if j.get("duration_months", 24) < 12)
    stability_penalty = min(short_stints * 0.08, 0.25)

    # 5. Education tier bonus
    edu_bonus = 0.0
    for edu in c.get("education", []):
        tier = edu.get("tier", "unknown")
        if tier == "tier_1":
            edu_bonus = max(edu_bonus, 0.10)
        elif tier == "tier_2":
            edu_bonus = max(edu_bonus, 0.05)

    # 6. Location fit
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    pref_locs = JD["preferred_locations"]
    pref_countries = JD["preferred_countries"]
    willing = c.get("redrob_signals", {}).get("willing_to_relocate", False)

    if any(pl in loc for pl in pref_locs):
        loc_s = 1.0
    elif country in pref_countries:
        loc_s = 0.7
    elif willing:
        loc_s = 0.5
    else:
        loc_s = 0.2

    # Combine
    raw = (
        yoe_s * 0.30
        + product_s * 0.30
        + title_s * 0.20
        + loc_s * 0.10
        + edu_bonus
        - stability_penalty
    )
    return max(0.0, min(raw, 1.0))


# ── Behavioral score (0-1) ───────────────────────────────────────────────────

def behavioral_score(c: Dict[str, Any]) -> float:
    sig = c.get("redrob_signals", {})

    # Recency: days since last active
    last_active_days = _days_since(sig.get("last_active_date", "2000-01-01"))
    if last_active_days < 14:
        recency_s = 1.0
    elif last_active_days < 30:
        recency_s = 0.85
    elif last_active_days < 60:
        recency_s = 0.65
    elif last_active_days < 90:
        recency_s = 0.45
    elif last_active_days < 180:
        recency_s = 0.25
    else:
        recency_s = 0.05

    # Open to work
    otw = 1.0 if sig.get("open_to_work_flag", False) else 0.4

    # Response rate
    rr = sig.get("recruiter_response_rate", 0.0)
    rr_s = rr  # already 0-1

    # Profile completeness
    pc = sig.get("profile_completeness_score", 0) / 100.0

    # Interview completion
    ic = sig.get("interview_completion_rate", 0.5)

    # Notice period (prefer < 30 days)
    notice = sig.get("notice_period_days", 90)
    if notice <= 30:
        notice_s = 1.0
    elif notice <= 60:
        notice_s = 0.6
    else:
        notice_s = 0.3

    # GitHub activity (only for tech roles)
    gh = sig.get("github_activity_score", -1)
    gh_s = (gh / 100.0) if gh >= 0 else 0.3  # unknown = neutral

    # Saved by recruiters (social proof)
    saved = min(sig.get("saved_by_recruiters_30d", 0), 20)
    saved_s = saved / 20.0

    # Verified signals
    verified = (sig.get("verified_email", False) + sig.get("verified_phone", False)) / 2.0

    raw = (
        recency_s * 0.25
        + otw * 0.15
        + rr_s * 0.20
        + pc * 0.10
        + ic * 0.10
        + notice_s * 0.10
        + gh_s * 0.05
        + saved_s * 0.03
        + verified * 0.02
    )
    return max(0.0, min(raw, 1.0))


# ── Honeypot detection ───────────────────────────────────────────────────────

def is_honeypot(c: Dict[str, Any]) -> bool:
    """
    Detect impossible / fake profiles. Returns True = remove from ranking.
    """
    profile = c.get("profile", {})
    skills = c.get("skills", [])
    history = c.get("career_history", [])
    sig = c.get("redrob_signals", {})

    # 1. Non-AI titles with suspiciously perfect AI skill list
    non_ai_titles = [
        "graphic designer", "marketing manager", "hr manager", "content writer",
        "accountant", "sales manager", "business development", "recruiter",
        "lawyer", "doctor",
    ]
    current_title = profile.get("current_title", "").lower()
    title_is_non_ai = any(t in current_title for t in non_ai_titles)

    ai_skills = [
        "embeddings", "transformers", "vector database", "nlp", "llm",
        "fine-tuning", "rag", "faiss", "pinecone", "ranking",
    ]
    ai_skill_count = sum(
        1 for s in skills
        if any(a in s.get("name", "").lower() for a in ai_skills)
    )
    skill_title_mismatch = title_is_non_ai and ai_skill_count >= 5

    # 2. Profile completeness too high with no career history detail
    pc = sig.get("profile_completeness_score", 0)
    total_career_text = sum(len(j.get("description", "")) for j in history)
    suspicious_completeness = pc > 95 and total_career_text < 100

    # 3. Impossible YoE vs career history
    yoe = profile.get("years_of_experience", 0)
    history_months = sum(j.get("duration_months", 0) for j in history)
    impossible_yoe = yoe > 0 and history_months > 0 and history_months < (yoe * 12 * 0.4)

    return skill_title_mismatch or suspicious_completeness


# ── Full feature extraction ───────────────────────────────────────────────────

def extract_features(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": c["candidate_id"],
        "skill_score": skill_score(c),
        "career_score": career_score(c),
        "behavior_score": behavioral_score(c),
        "is_honeypot": is_honeypot(c),
        "text": _build_candidate_text(c),
        # Extra metadata for reasoning
        "current_title": c.get("profile", {}).get("current_title", ""),
        "yoe": c.get("profile", {}).get("years_of_experience", 0),
        "location": c.get("profile", {}).get("location", ""),
        "response_rate": c.get("redrob_signals", {}).get("recruiter_response_rate", 0),
        "open_to_work": c.get("redrob_signals", {}).get("open_to_work_flag", False),
        "notice_days": c.get("redrob_signals", {}).get("notice_period_days", 90),
        "skill_count": len(c.get("skills", [])),
    }

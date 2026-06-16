"""
parse_jd.py — Extract structured requirements from the job description.
Hard-coded for this specific JD so Phase 2 ranking is deterministic.
"""

JD_PROFILE = {
    # ── Skills ──────────────────────────────────────────────────────────────
    "must_have_skills": [
        # embeddings / retrieval
        "embeddings", "sentence-transformers", "vector search", "vector database",
        "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
        "opensearch", "dense retrieval", "hybrid search", "semantic search",
        "information retrieval",
        # ranking / evaluation
        "ranking", "learning to rank", "ndcg", "mrr", "a/b testing",
        "recommendation system", "reranking", "re-ranking",
        # core ML / NLP
        "nlp", "natural language processing", "transformers", "bert",
        "machine learning", "deep learning", "python",
        # IR building blocks
        "bm25", "tf-idf", "tfidf",
    ],
    "nice_to_have_skills": [
        "lora", "qlora", "peft", "fine-tuning", "fine tuning", "llm",
        "large language model", "rag", "retrieval augmented generation",
        "xgboost", "lightgbm", "distributed systems", "pytorch", "tensorflow",
        "mlops", "model serving", "inference optimization",
        "open source", "github", "hr-tech", "recruiting",
    ],
    "anti_skills": [
        # CV/vision-primary (no NLP signal)
        "computer vision", "image classification", "object detection",
        "image segmentation", "photoshop", "illustrator",
        # speech-primary
        "speech recognition", "tts", "text to speech", "asr",
        # robotics
        "robotics", "ros",
        # pure frontend / unrelated
        "tailwind", "react", "vue", "angular", "figma",
        # consulting red flags (skill names sometimes hint)
    ],

    # ── Experience ──────────────────────────────────────────────────────────
    "yoe_min": 5,
    "yoe_max": 9,
    "yoe_ideal_min": 6,
    "yoe_ideal_max": 8,

    # ── Location ────────────────────────────────────────────────────────────
    "preferred_locations": [
        "pune", "noida", "delhi", "ncr", "delhi ncr", "hyderabad",
        "mumbai", "bengaluru", "bangalore", "gurgaon", "gurugram",
    ],
    "preferred_countries": ["india"],

    # ── Career quality signals ───────────────────────────────────────────────
    "product_company_industries": [
        "software", "technology", "saas", "fintech", "edtech", "healthtech",
        "e-commerce", "marketplace", "ai", "ml", "data", "internet",
        "consumer internet", "enterprise software",
    ],
    "anti_company_types": [
        # pure IT services / consulting (whole-career disqualifier)
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "hcl", "tech mahindra", "mphasis", "hexaware",
    ],

    # ── Notice period ────────────────────────────────────────────────────────
    "notice_soft_cap_days": 30,

    # ── Scoring weights (must sum to 1.0) ────────────────────────────────────
    "weights": {
        "skill": 0.40,
        "career": 0.35,
        "tfidf": 0.15,
        "behavior": 0.10,
    },
}


def get_jd_profile():
    return JD_PROFILE


def get_jd_text():
    """Full JD as a single string for TF-IDF query vector."""
    return (
        "Senior AI Engineer founding team embeddings retrieval ranking NLP "
        "transformers vector database faiss pinecone weaviate elasticsearch "
        "semantic search hybrid search learning to rank NDCG evaluation "
        "recommendation system fine-tuning LLM RAG Python production "
        "product company startup applied ML 5 to 9 years experience "
        "Pune Noida India"
    )

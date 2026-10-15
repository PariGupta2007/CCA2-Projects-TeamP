"""
CCA-2 (AI ML 1) Project
Team Members:
1. Pari Gupta - PRN: 1302250444
2. Manjiri Bankhele - PRN: 1302250623
Description: Part 3 - Retrieval-Augmented Generation (RAG) Guardrail Framework
"""

import json

# PERSISTENT CONTEXT DOMAIN REPOSITORY
INTERNAL_KNOWLEDGE_BASE = {
    "company_policy": "Pari-Manjiri Corp allows maximum 15 days of annual paid leave. Remote work requests require a 48-hour notice window.",
    "server_guide": "The secure production server is hosted at IP address 192.168.45.2. Access requires verified VPN authentication tokens.",
    "refund_policy": "Product refunds are strictly processed within 30 days of initial purchase, provided the product remains unopened."
}

def minimal_rag_engine(user_query):
    query_tokens = set(user_query.lower().split())
    best_match_key = None
    highest_relevance_score = 0.0
    
    # Parsing document vectors to score token overlap density
    for key, doc_text in INTERNAL_KNOWLEDGE_BASE.items():
        doc_tokens = set(doc_text.lower().replace('.', '').replace(',', '').split())
        token_intersection = query_tokens.intersection(doc_tokens)
        
        relevance_score = len(token_intersection) / max(len(query_tokens), 1)
        
        if relevance_score > highest_relevance_score:
            highest_relevance_score = relevance_score
            best_match_key = key
            
    # BOUNDARY FALLBACK GUARDRAILS
    # Rejects baseline hallucination vectors if intersection drops below 15% threshold
    RELEVANCE_THRESHOLD = 0.15
    
    if highest_relevance_score < RELEVANCE_THRESHOLD or not best_match_key:
        response_payload = {
            "status": "error",
            "retrieved_context": None,
            "relevance_score": round(highest_relevance_score, 4),
            "answer": "Information not found in database"
        }
    else:
        response_payload = {
            "status": "success",
            "retrieved_context": f"Matched Source Module: [{best_match_key}]",
            "relevance_score": round(highest_relevance_score, 4),
            "answer": INTERNAL_KNOWLEDGE_BASE[best_match_key]
        }
        
    return json.dumps(response_payload, indent=4)

# =====================================================================
# RUNNING SIMULATED ENGINE VALIDATIONS
# =====================================================================
print("---------------------------------------------------------------------")
print("VERIFICATION TASK 1: In-Scope Target Query Processing")
print("---------------------------------------------------------------------")
query_in_scope = "What is the policy regarding remote work notice times?"
print(f"User Query : {query_in_scope}")
print(f"Engine Out :\n{minimal_rag_engine(query_in_scope)}\n")

print("---------------------------------------------------------------------")
print("VERIFICATION TASK 2: Out-Of-Scope Query Handling (Guardrail Test)")
print("---------------------------------------------------------------------")
query_out_of_scope = "What is the current stock price of Apple or Microsoft today?"
print(f"User Query : {query_out_of_scope}")
print(f"Engine Out :\n{minimal_rag_engine(query_out_of_scope)}")
print("---------------------------------------------------------------------")

"""
CCA-2 (AI ML 1) Project
Team Members:
1. Pari Gupta - PRN: 1302250444
2. Manjiri Bankhele - PRN: 1302250623
Description: Part 2 - Hugging Face Sentiment Monitor Application
"""

from transformers import pipeline

print(" Loading Hugging Face Pipeline Architecture (distilbert-base)...")
classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

# ---------------------------------------------------------------------
# USE CASE 1: CUSTOMER SUPPORT URGENCY ROUTER
# ---------------------------------------------------------------------
def assess_ticket_urgency(ticket_text):
    safety_triggers = ["fire", "smoke", "exploded", "burn", "shocked", "injury"]
    if any(word in ticket_text.lower() for word in safety_triggers):
        return "[CRITICAL] Safety Override Initiated (Hazard Detected)"
    
    inference = classifier(ticket_text)[0]
    if inference['label'] == 'NEGATIVE' and inference['score'] > 0.85:
        return "[HIGH] Urgent Priority Queue"
    return "[LOW] Standard Operational Queue"

# ---------------------------------------------------------------------
# USE CASE 2: GRANULAR PRODUCT REVIEW ANALYZER
# ---------------------------------------------------------------------
def analyze_product_review(review_text):
    inference = classifier(review_text)[0]
    score = inference['score']
    
    # Mathematical threshold integration mapping confidence scores to Neutral vectors
    if 0.50 <= score <= 0.70:
        return f"NEUTRAL | Score Confidence: {score:.2%}"
    elif inference['label'] == 'POSITIVE':
        return f"POSITIVE | Score Confidence: {score:.2%}"
    else:
        return f"NEGATIVE | Score Confidence: {score:.2%}"

# ---------------------------------------------------------------------
# USE CASE 3: FINANCIAL NEWS INDEXER
# ---------------------------------------------------------------------
def analyze_financial_headline(headline):
    bearish_lexicon = ["plummet", "crash", "investigation", "bankrupt", "deficit"]
    if any(word in headline.lower() for word in bearish_lexicon):
        return "BEARISH MARKET OUTLOOK"
        
    inference = classifier(headline)[0]
    if inference['label'] == 'POSITIVE':
        return "BULLISH MARKET OUTLOOK"
    return "BEARISH MARKET OUTLOOK"

# =====================================================================
# INTEGRATION TESTING MATRIX
# =====================================================================
print("\n" + "="*70)
print("MODULE 1: CUSTOMER TICKETING ROUTING PIPELINE")
print("="*70)
tickets = [
    "My router caught fire and filled the living room with smoke!",
    "The internet drops for 5 minutes every day at midnight. Annoying.",
]
for t in tickets:
    print(f"Input Text : \"{t}\"\nRouting    : {assess_ticket_urgency(t)}\n")

print("="*70)
print("MODULE 2: CUSTOMER FEEDBACK BREAKDOWN MONITOR")
print("="*70)
reviews = [
    "It works okay, but the battery dies fast.",
    "Absolutely amazing product! Best purchase of the entire year.",
]
for r in reviews:
    print(f"Input Text : \"{r}\"\nEvaluation : {analyze_product_review(r)}\n")

print("="*70)
print("MODULE 3: FINANCIAL NEWS MACRO PIPELINE")
print("="*70)
headlines = [
    "Company X profits double this quarter exceeding all expectations!",
    "Regulatory investigation launched against tech giants over data deficits.",
]
for h in headlines:
    print(f"Input Text : \"{h}\"\nSentiment  : {analyze_financial_headline(h)}\n")
print("="*70)

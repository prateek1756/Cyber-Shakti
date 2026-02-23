"""
Fraud Message Detection API
AI-powered fraud detection using NLP and pattern matching
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from typing import Dict, List, Tuple
from model_manager import model_manager

app = Flask(__name__)
CORS(app)

# Fraud keywords and patterns
# Categories of fraud for more specific detection
SCAM_CATEGORIES = {
    'financial': ['bank', 'account', 'credit card', 'payment', 'transfer', 'wire', 'transaction', 'balance', 'limit', 'overdrawn', 'loan', 'mortgage'],
    'identity': ['ssn', 'social security', 'identity', 'passport', 'driver license', 'dob', 'birth date', 'verification', 'security question'],
    'urgency_threat': ['suspended', 'deleted', 'blocked', 'arrest', 'warrant', 'police', 'irs', 'government', 'legal action', 'immediately', 'urgent', 'now'],
    'prize_lottery': ['winner', 'won', 'lottery', 'prize', 'gift card', 'congratulations', 'claim', 'award', 'vacation', 'free', 'million', 'dollars'],
    'crypto_investment': ['bitcoin', 'crypto', 'investment', 'profit', 'trading', 'binance', 'wallet', 'seed phrase', 'return', 'guaranteed', 'passive income'],
    'job_offer': ['hiring', 'salary', 'working from home', 'remote', 'interview', 'hr', 'application', 'offer', 'part-time', 'full-time'],
    'tech_support': ['microsoft', 'apple', 'amazon', 'support', 'technician', 'virus', 'malware', 'security alert', 're-verify', 'system compromised']
}

# Fraud keywords with weighted scores
FRAUD_KEYWORDS = {
    'urgent': 3, 'immediately': 3, 'verify': 2, 'suspended': 4, 'account': 2, 'click': 2, 
    'password': 4, 'otp': 5, 'pin': 5, 'bank': 3, 'credit card': 4, 'social security': 5,
    'ssn': 5, 'prize': 3, 'winner': 4, 'lottery': 5, 'claim': 2, 'refund': 3, 'tax': 2,
    'irs': 4, 'inheritance': 5, 'wire transfer': 5, 'bitcoin': 3, 'gift card': 5,
    'limited time': 3, 'act now': 3, 'final notice': 5, 'arrest': 6, 'warrant': 6,
    'overdue': 3, 'seed phrase': 10, 'recovery key': 8, 'unauthorized': 4,
    'login attempt': 4, 'abnormal activity': 4, 'kindly': 2, 'dear customer': 3,
    'valuable customer': 3, 'dear user': 2
}

# Suspicious URL patterns
SUSPICIOUS_URL_PATTERNS = [
    r'bit\.ly',
    r'tinyurl',
    r'goo\.gl',
    r't\.co',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP addresses
    r'[a-z0-9-]+\.tk',  # Free domains
    r'[a-z0-9-]+\.ml',
    r'[a-z0-9-]+\.ga',
    r'[a-z0-9-]+\.cf',
    r'[a-z0-9-]+\.gq',
]

# Urgency patterns
URGENCY_PATTERNS = [
    r'within \d+ (hours?|minutes?|days?)',
    r'expires? (today|tonight|soon)',
    r'act (now|immediately|fast)',
    r'limited time',
    r'hurry',
    r'don\'t (wait|delay)',
]

# Request for sensitive info patterns
SENSITIVE_INFO_PATTERNS = [
    r'(enter|provide|confirm|verify).{0,20}(password|pin|otp|ssn|social security)',
    r'(click|tap).{0,30}(link|here|below)',
    r'reply.{0,20}(with|your).{0,20}(password|pin|otp|code)',
]


def analyze_keywords(message: str) -> Tuple[int, List[str]]:
    """Analyze message for fraud keywords"""
    message_lower = message.lower()
    score = 0
    matches = []
    
    for keyword, weight in FRAUD_KEYWORDS.items():
        if keyword in message_lower:
            score += weight
            matches.append(keyword)
    
    # Normalize score to 0-100
    max_possible = 50  # Reasonable max for keyword matching
    normalized_score = min(100, int((score / max_possible) * 100))
    
    return normalized_score, matches


def analyze_urls(message: str) -> Tuple[int, List[str]]:
    """Analyze URLs in message"""
    suspicious_urls = []
    
    # Find all URLs
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, message)
    
    for url in urls:
        for pattern in SUSPICIOUS_URL_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                suspicious_urls.append(url)
                break
    
    # Score based on number of suspicious URLs
    score = min(100, len(suspicious_urls) * 40)
    
    return score, suspicious_urls


def analyze_urgency(message: str) -> int:
    """Analyze urgency indicators"""
    score = 0
    
    for pattern in URGENCY_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            score += 20
    
    return min(100, score)


def analyze_sensitive_requests(message: str) -> int:
    """Analyze requests for sensitive information"""
    score = 0
    
    for pattern in SENSITIVE_INFO_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            score += 30
    
    return min(100, score)


def simple_nlp_fraud_score(message: str) -> Tuple[float, str, float]:
    """
    NLP-based fraud detection using Scikit-Learn model
    Returns: (fraud_score, label, confidence)
    """
    return model_manager.predict(message)


def analyze_categories(message: str) -> Dict[str, int]:
    """Calculate scores for different scam categories"""
    message_lower = message.lower()
    cat_scores = {}
    
    for cat, keywords in SCAM_CATEGORIES.items():
        matches = [kw for kw in keywords if kw in message_lower]
        score = min(100, int((len(matches) / (len(keywords) * 0.3) if keywords else 0) * 100)) if matches else 0
        cat_scores[cat] = score
        
    return cat_scores


def analyze_human_vs_phishing(message: str) -> Dict:
    """Analyze if the message looks like scripted phishing or natural human text"""
    score = 0
    indicators = []
    
    # Phishing indicators
    if re.search(r'^[A-Z\s!]+$', message) and len(message) > 10:
        score += 30
        indicators.append("Aggressive all-caps style")
        
    if "kindly" in message.lower() or "please do not hesitate" in message.lower():
        score += 20
        indicators.append("Formal/Polite scam-typical vocabulary")
        
    if re.search(r'customer|user|member', message.lower()) and not re.search(r'[A-Z][a-z]+', message):
        # Generic greetings without specific names
        score += 25
        indicators.append("Generic non-personalized greeting")
        
    # Check for templated patterns (e.g. "Your [X] has been [Y]")
    if re.search(r'your [a-z]+ has been [a-z]+', message.lower()):
        score += 20
        indicators.append("Templated alert structure")

    # Human flow indicators (would subtract from score)
    # Human text often has contractions, varying sentence length, typos, slang
    human_indicators = 0
    if re.search(r"i'm|don't|can't|won't|it's", message.lower()):
        human_indicators += 15
    if len(message.split()) < 10:
        human_indicators += 10 # Short messages are often human
        
    score = max(0, score - human_indicators)
    
    return {
        "score": min(100, score),
        "indicators": indicators
    }


def detect_fraud(message: str) -> Dict:
    """Main fraud detection function"""
    
    # Run all analyses
    keyword_score, keyword_matches = analyze_keywords(message)
    url_score, suspicious_urls = analyze_urls(message)
    urgency_score = analyze_urgency(message)
    sensitive_score = analyze_sensitive_requests(message)
    nlp_score, nlp_label, nlp_confidence = simple_nlp_fraud_score(message)
    
    cat_scores = analyze_categories(message)
    phishing_analysis = analyze_human_vs_phishing(message)
    
    # Top scam category
    top_category = max(cat_scores.items(), key=lambda x: x[1]) if any(cat_scores.values()) else ("none", 0)
    
    # Calculate overall risk score (weighted average)
    risk_score = int(
        (nlp_score * 0.3) +
        (phishing_analysis['score'] * 0.2) +
        (keyword_score * 0.2) +
        (url_score * 0.15) +
        (urgency_score * 0.1) +
        (sensitive_score * 0.05)
    )
    
    # Determine classification
    classification = "fraud" if risk_score >= 50 else "safe"
    if 30 <= risk_score < 50:
        classification = "suspicious"
    
    # Generate explanations
    explanations = []
    
    if nlp_score >= 60:
        explanations.append(f"AI patterns match known fraud signatures ({nlp_score:.0f}%)")
    
    if phishing_analysis['indicators']:
        explanations.append(f"Heuristics: {phishing_analysis['indicators'][0]}")
    
    if top_category[1] >= 40:
        explanations.append(f"Matches pattern for {top_category[0].replace('_', ' ')} scam")
    
    if keyword_matches:
        top_keywords = keyword_matches[:2]
        explanations.append(f"Suspicious words: {', '.join(top_keywords)}")
    
    if suspicious_urls:
        explanations.append(f"Detected {len(suspicious_urls)} high-risk link(s)")
    
    if not explanations:
        explanations.append("Message appears to be normal communication")
    
    return {
        "classification": classification,
        "risk_score": risk_score,
        "explanations": explanations,
        "details": {
            "nlp": {
                "fraud_score": round(nlp_score, 1),
                "label": nlp_label,
                "confidence": round(nlp_confidence, 2)
            },
            "phishing_heuristics": phishing_analysis,
            "category_analysis": cat_scores,
            "top_category": top_category[0],
            "keywords": {
                "score": keyword_score,
                "matches": keyword_matches[:5]
            },
            "urls": {
                "score": url_score,
                "suspicious_urls": suspicious_urls
            },
            "urgency_score": urgency_score,
            "sensitive_info_score": sensitive_score
        }
    }


@app.route('/detect', methods=['POST'])
def detect():
    """Fraud detection endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Missing 'message' field in request"
            }), 400
        
        message = data['message']
        
        if not message or not message.strip():
            return jsonify({
                "error": "Message cannot be empty"
            }), 400
        
        # Perform fraud detection
        result = detect_fraud(message)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "fraud-detection-api"
    }), 200


if __name__ == '__main__':
    print("=" * 60)
    print("Fraud Message Detection API")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("Endpoints:")
    print("  POST /detect - Analyze message for fraud")
    print("  GET  /health - Health check")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8000, debug=True)

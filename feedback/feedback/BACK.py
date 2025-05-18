from flask import Flask, render_template, request, jsonify
from datetime import datetime
import spacy
import re
import json
from collections import Counter
import os
import uuid

# Initialize Flask app
app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if model isn't installed
    import subprocess
    subprocess.call(["/usr/local/bin/python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Feedback storage file
FEEDBACK_STORAGE = 'data/feedback_history.json'

# Load existing feedback if available
def load_feedback_history():
    if os.path.exists(FEEDBACK_STORAGE):
        try:
            with open(FEEDBACK_STORAGE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

# Save feedback history
def save_feedback_history(history):
    with open(FEEDBACK_STORAGE, 'w') as f:
        json.dump(history, f, indent=2)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze feedback and return the results"""
    data = request.get_json()
    
    # Extract data from request
    feedback_text = data.get("text", "").strip()
    feedback_reason = data.get("reason", "")
    rating = data.get("rating", 0)
    
    # Basic validation
    if not feedback_text or len(feedback_text) < 10:
        return jsonify({"error": "Feedback text is too short"}), 400
    
    if not feedback_reason:
        return jsonify({"error": "Feedback reason is required"}), 400
    
    if rating < 1 or rating > 5:
        return jsonify({"error": "Invalid rating value"}), 400
    
    # Analyze the feedback
    analysis_results = analyze_feedback(feedback_text, feedback_reason, rating)
    
    # Create a feedback record
    feedback_record = {
        "id": str(uuid.uuid4()),
        "reason": feedback_reason,
        "text": feedback_text,
        "rating": rating,
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis_results
    }
    
    # Add to history
    history = load_feedback_history()
    history.insert(0, feedback_record)  # Add to the beginning
    
    # Limit history to 50 items
    if len(history) > 50:
        history = history[:50]
    
    # Save updated history
    save_feedback_history(history)
    
    return jsonify({
        "success": True,
        "feedback_id": feedback_record["id"],
        "analysis": analysis_results
    })

@app.route("/history", methods=["GET"])
def get_history():
    """Get feedback history"""
    history = load_feedback_history()
    return jsonify(history)

@app.route("/history/<feedback_id>", methods=["GET"])
def get_feedback_item(feedback_id):
    """Get a specific feedback item"""
    history = load_feedback_history()
    
    for item in history:
        if item["id"] == feedback_id:
            return jsonify(item)
    
    return jsonify({"error": "Feedback item not found"}), 404

@app.route("/history/<feedback_id>", methods=["DELETE"])
def delete_feedback_item(feedback_id):
    """Delete a specific feedback item"""
    history = load_feedback_history()
    
    # Filter out the item to delete
    new_history = [item for item in history if item["id"] != feedback_id]
    
    if len(new_history) < len(history):
        save_feedback_history(new_history)
        return jsonify({"success": True})
    
    return jsonify({"error": "Feedback item not found"}), 404

@app.route("/history", methods=["DELETE"])
def clear_history():
    """Clear all feedback history"""
    save_feedback_history([])
    return jsonify({"success": True})

def analyze_feedback(text, reason, rating):
    """Analyze feedback text and return comprehensive analysis"""
    # Process text with spaCy
    doc = nlp(text)
    
    # Sentiment analysis
    sentiment_score = calculate_sentiment_score(doc, text, rating)
    sentiment_type = categorize_sentiment(sentiment_score)
    
    # Grammar check
    grammar_errors, readability_score = check_grammar(doc, text)
    
    # Specificity analysis
    specificity_score, specificity_level = analyze_specificity(doc, text)
    
    # Politeness analysis
    politeness_score, politeness_level = analyze_politeness(doc, text)
    
    # Temporal context
    temporal_status, temporal_expressions = analyze_temporal_context(doc, text)
    
    # Relevance to issue
    relevance_score, relevance_level = analyze_relevance(doc, text, reason)
    
    # Calculate validation confidence
    validation_confidence = calculate_validation_confidence(
        sentiment_score, specificity_score, politeness_score, relevance_score
    )
    
    return {
        "valid": validation_confidence > 60,
        "confidence": f"{validation_confidence:.1f}",
        "sentiment": {
            "type": sentiment_type,
            "score": f"{sentiment_score:.2f}"
        },
        "grammar": {
            "errors": grammar_errors,
            "readability": f"{readability_score:.1f}"
        },
        "specificity": {
            "level": specificity_level,
            "score": f"{specificity_score:.1f}"
        },
        "politeness": {
            "level": politeness_level,
            "score": f"{politeness_score:.1f}"
        },
        "temporal": {
            "status": temporal_status,
            "expressions": temporal_expressions
        },
        "relevance": {
            "level": relevance_level,
            "score": f"{relevance_score:.1f}"
        }
    }

def calculate_sentiment_score(doc, text, rating):
    """Calculate sentiment score based on text and rating"""
    # Base sentiment from rating (0-1 scale)
    if rating >= 4:
        base_score = 0.7 + (rating - 4) * 0.15  # 0.7-1.0
    elif rating <= 2:
        base_score = 0.3 - (2 - rating) * 0.15  # 0.0-0.3
    else:
        base_score = 0.5  # Neutral for rating 3
    
    # Adjust based on positive and negative words
    positive_words = [
        'great', 'good', 'excellent', 'love', 'like', 'appreciate', 'thank',
        'helpful', 'fantastic', 'amazing', 'brilliant', 'wonderful', 'pleased'
    ]
    
    negative_words = [
        'bad', 'poor', 'terrible', 'hate', 'dislike', 'awful', 'worst',
        'horrible', 'disappointed', 'useless', 'frustrating', 'annoying', 'issue'
    ]
    
    # Tokenize and lowercase for word analysis
    words = [token.text.lower() for token in doc]
    
    # Count positive and negative words
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Adjust score based on word count
    word_adjustment = (positive_count - negative_count) * 0.05
    
    # Final score clamped between 0 and 1
    final_score = max(0, min(1, base_score + word_adjustment))
    
    return final_score

def categorize_sentiment(score):
    """Categorize sentiment score into NEGATIVE, NEUTRAL, or POSITIVE"""
    if score < 0.4:
        return "NEGATIVE"
    elif score > 0.6:
        return "POSITIVE"
    else:
        return "NEUTRAL"

def check_grammar(doc, text):
    """Check grammar and calculate readability score"""
    # Simple grammar check (count potential errors)
    
    # Count sentence fragments (sentences without verbs)
    sentences = [sent for sent in doc.sents]
    fragments = sum(1 for sent in sentences if not any(token.pos_ == "VERB" for token in sent))
    
    # Count potential spelling errors (out-of-vocabulary words)
    spelling_errors = sum(1 for token in doc if token.is_alpha and not token.is_stop and not token.is_punct and not token.is_space and token.vocab[token.text].is_oov)
    
    # Count repeated words
    words = [token.text.lower() for token in doc if token.is_alpha]
    word_counts = Counter(words)
    repeated_words = sum(1 for word, count in word_counts.items() if count > 3 and word not in ["the", "a", "an", "and", "or", "but", "in", "on", "with", "to", "of"])
    
    # Total estimated errors
    total_errors = fragments + spelling_errors + min(repeated_words, 2)
    
    # Calculate readability score (simplified Flesch-Kincaid)
    # Higher is better/more readable
    word_count = len([token for token in doc if token.is_alpha])
    if word_count == 0:
        return total_errors, 50.0
    
    sentence_count = len(sentences)
    if sentence_count == 0:
        sentence_count = 1
    
    avg_words_per_sentence = word_count / sentence_count
    avg_syllables_per_word = 1.5  # Simplified estimate
    
    # Simplified Flesch Reading Ease score
    readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
    
    # Clamp between 0-100
    readability = max(0, min(100, readability))
    
    return total_errors, readability
def analyze_specificity(doc, text):
    """Analyze the specificity of the feedback"""
    # Start with base score
    score = 5.0
    
    # Length affects specificity
    if len(text) > 200:
        score += 2.0
    elif len(text) > 100:
        score += 1.0
    
    # Check for specificity indicators
    specificity_indicators = [
        'specifically', 'particular', 'exactly', 'precisely', 'detail',
        'example', 'instance', 'case', 'scenario', 'situation', 'occurred',
        'happened', 'experienced', 'noticed', 'observed', 'specific'
    ]
    
    # Count specificity indicators
    indicator_count = sum(1 for token in doc if token.text.lower() in specificity_indicators)
    score += min(indicator_count * 0.5, 2.0)
    
    # Check for numbers (dates, times, quantities)
    number_count = sum(1 for token in doc if token.like_num)
    score += min(number_count * 0.3, 1.0)
    
    # Named entities indicate specificity
    entity_count = len(doc.ents)
    score += min(entity_count * 0.3, 1.0)
    
    # Clamp score between 1-10
    final_score = max(1.0, min(10.0, score))
    
    # Determine level
    if final_score < 4.0:
        level = "LOW"
    elif final_score > 7.0:
        level = "HIGH"
    else:
        level = "MEDIUM"
    
    return final_score, level

def analyze_politeness(doc, text):
    """Analyze the politeness level of the feedback"""
    # Start with neutral score
    score = 3.0
    
    # Check for polite phrases and words
    polite_indicators = [
        'please', 'thank', 'appreciate', 'grateful', 'kind', 'would',
        'could', 'may', 'might', 'consider', 'suggest', 'perhaps', 'possibly'
    ]
    
    # Check for impolite phrases and words
    impolite_indicators = [
        'stupid', 'dumb', 'hate', 'terrible', 'worst', 'awful', 'horrible',
        'useless', 'incompetent', 'ridiculous', 'pathetic', 'disaster'
    ]
    
    # Count indicators
    polite_count = sum(1 for token in doc if token.text.lower() in polite_indicators)
    impolite_count = sum(1 for token in doc if token.text.lower() in impolite_indicators)
    
    # Adjust score
    score += polite_count * 0.5
    score -= impolite_count * 0.5
    
    # Clamp score between 1-5
    final_score = max(1.0, min(5.0, score))
    
    # Determine level
    if final_score < 2.5:
        level = "IMPOLITE"
    elif final_score > 3.5:
        level = "POLITE"
    else:
        level = "NEUTRAL"
    
    return final_score, level

def analyze_temporal_context(doc, text):
    """Analyze temporal context in the feedback"""
    # Temporal words and phrases to look for
    temporal_indicators = [
        'yesterday', 'today', 'tomorrow', 'last week', 'next week',
        'last month', 'next month', 'recently', 'currently', 'now',
        'previously', 'earlier', 'later', 'ago', 'since', 'before',
        'after', 'during', 'while', 'when', 'whenever', 'always',
        'never', 'sometimes', 'often', 'rarely', 'usually'
    ]
    
    # Find temporal expressions in text
    found_expressions = []
    
    # Look for direct matches
    text_lower = text.lower()
    for indicator in temporal_indicators:
        if indicator in text_lower:
            found_expressions.append(indicator)
    
    # Also check for dates using spaCy's entities
    date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    found_expressions.extend(date_entities)
    
    # Remove duplicates and sort
    found_expressions = sorted(set(found_expressions))
    
    # Determine status
    if found_expressions:
        status = "Present"
        expressions = ", ".join(found_expressions)
    else:
        status = "Absent"
        expressions = "none"
    
    return status, expressions

def analyze_relevance(doc, text, reason):
    """Analyze the relevance of feedback to the selected reason"""
    # Define keywords for each reason type
    reason_keywords = {
        "connection_issues": [
            "connection", "disconnect", "dropped", "unstable", "internet",
            "wifi", "network", "bandwidth", "latency", "lag", "delay"
        ],
        "audio_quality": [
            "audio", "sound", "hear", "volume", "microphone", "speaker",
            "echo", "noise", "mute", "voice", "speech", "silent"
        ],
        "video_quality": [
            "video", "camera", "webcam", "resolution", "blur", "freeze",
            "pixelated", "dark", "bright", "lighting", "background", "visual"
        ],
        "meeting_experience": [
            "meeting", "call", "conference", "session", "experience", "interface",
            "join", "invite", "schedule", "calendar", "notification", "reminder"
        ],
        "feature_request": [
            "feature", "add", "implement", "improve", "enhance", "function",
            "capability", "option", "setting", "preference", "tool", "integration"
        ]
    }
    
    # Base score
    base_score = 7.0
    
    # Get keywords for the selected reason
    keywords = reason_keywords.get(reason, [])
    
    if keywords:
        # Count keyword matches
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Adjust score based on matches
        if matches >= 3:
            base_score += 2.0
        elif matches >= 1:
            base_score += 1.0
        else:
            base_score -= 1.0
    
    # Length can indicate relevance (very short feedback is often less relevant)
    if len(text) < 50:
        base_score -= 1.0
    
    # Clamp score between 1-10
    final_score = max(1.0, min(10.0, base_score))
    
    # Determine level
    if final_score < 5.0:
        level = "LOW"
    elif final_score < 8.0:
        level = "MEDIUM"
    else:
        level = "HIGH"
    
    return final_score, level

def calculate_validation_confidence(sentiment_score, specificity_score, politeness_score, relevance_score):
    """Calculate overall validation confidence score"""
    # Weight factors
    sentiment_weight = 0.15
    specificity_weight = 0.35
    politeness_weight = 0.15
    relevance_weight = 0.35
    
    # Calculate weighted score
    weighted_score = (
        (sentiment_score if sentiment_score > 0.3 else sentiment_score * 0.5) * sentiment_weight +
        (specificity_score / 10) * specificity_weight +
        (politeness_score / 5) * politeness_weight +
        (relevance_score / 10) * relevance_weight
    ) * 100
    
    # Return confidence percentage
    return weighted_score

if __name__ == "__main__":
    app.run(debug=True)
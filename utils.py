import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    """Clean and normalize text input."""
    if not text:
        return ""
    text_lower = text.lower()
    text_cleaned = re.sub(r"[^a-z0-9 ]", " ", text_lower)
    text_final = " ".join(text_cleaned.split())
    return text_final

def calculate_text_stats(problem_text, input_text, output_text):
    """Calculate statistics about the problem text."""
    full_text = problem_text + " " + input_text + " " + output_text
    words = full_text.split()
    word_count = len(words)
    char_count = len(full_text)
    
    # Look for complexity-related terms
    complexity_terms = [
        'algorithm', 'optimize', 'complexity', 'efficient', 
        'dynamic', 'graph', 'tree', 'recursion', 'backtrack',
        'greedy', 'divide', 'conquer', 'memoization'
    ]
    found_terms = []
    text_lower = full_text.lower()
    for term in complexity_terms:
        if term in text_lower:
            found_terms.append(term)
    
    # Count mathematical symbols
    math_symbols = ['$', '≤', '≥', '∑', '∏', '∫', '∞', '∈', '∉']
    math_count = sum(full_text.count(symbol) for symbol in math_symbols)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'complexity_indicators': len(found_terms),
        'math_symbols': math_count,
        'found_keywords': found_terms[:6]
    }

def find_most_similar(user_text, vectorizer, problems_df, top_k=5):
    """Find most similar problems from the dataset."""
    if problems_df is None or problems_df.empty:
        return []
    
    user_cleaned = clean_text(user_text)
    if not user_cleaned:
        return []
    
    try:
        user_vector = vectorizer.transform([user_cleaned])
        
        # Prepare problem texts
        problem_texts_list = []
        for idx, row in problems_df.iterrows():
            desc = str(row.get('description', ''))
            inp = str(row.get('input_description', ''))
            out = str(row.get('output_description', ''))
            combined = desc + " " + inp + " " + out
            problem_texts_list.append(clean_text(combined))
        
        if not problem_texts_list:
            return []
        
        problem_vectors = vectorizer.transform(problem_texts_list)
        
        # Compute similarities
        similarity_scores = cosine_similarity(user_vector, problem_vectors)[0]
        
        # Get top matches
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        similar_list = []
        
        for idx in top_indices:
            similarity_value = similarity_scores[idx]
            if similarity_value > 0.1:
                row = problems_df.iloc[idx]
                similar_list.append({
                    'title': row.get('title', 'Unknown'),
                    'similarity': similarity_value,
                    'class': row.get('problem_class', 'Unknown'),
                    'score': row.get('problem_score', 0),
                    'url': row.get('url', '')
                })
        
        return similar_list
    except Exception as e:
        return []

def get_prediction_details(classifier, regressor, text_vector):
    """Get prediction with confidence scores."""
    predicted_class = classifier.predict(text_vector)[0]
    predicted_score = regressor.predict(text_vector)[0]
    
    # Get confidence if available
    if hasattr(classifier, 'predict_proba'):
        class_probabilities = classifier.predict_proba(text_vector)[0]
        class_labels = classifier.classes_
        max_confidence = max(class_probabilities) * 100
        prob_dict = {class_labels[i]: class_probabilities[i] * 100 
                    for i in range(len(class_labels))}
    else:
        max_confidence = 85.0
        prob_dict = {predicted_class: max_confidence}
    
    return predicted_class, predicted_score, max_confidence, prob_dict

def format_difficulty_display(difficulty_class):
    """Format difficulty class with emoji."""
    difficulty_lower = difficulty_class.lower()
    emoji_map = {
        "easy": "🟢",
        "medium": "🟡",
        "hard": "🔴"
    }
    return emoji_map.get(difficulty_lower, "⚪")

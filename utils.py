import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

def clean_text(text):
    """Clean and normalize text for processing"""
    if not text:
        return ""
    text_lower = text.lower()
    text_cleaned = re.sub(r"[^a-z0-9 ]", " ", text_lower)
    text_final = " ".join(text_cleaned.split())
    return text_final

def extract_features_from_text(problem_text, input_text, output_text, problem_score=None):
    """
    Extract numerical features from problem text
    This should match the training feature extraction
    """
    text = (problem_text + ' ' + input_text + ' ' + output_text).lower()
    
    features = {}
    
    if problem_score is not None:
        features['problem_score'] = problem_score
        features['score_squared'] = problem_score ** 2
        features['score_cubed'] = problem_score ** 3
    else:
        features['problem_score'] = 5.0
        features['score_squared'] = 25.0
        features['score_cubed'] = 125.0

    features['total_length'] = len(text)
    features['word_count'] = len(text.split())
    features['desc_length'] = len(problem_text)
    features['input_desc_length'] = len(input_text)
    features['output_desc_length'] = len(output_text)
    
    words = text.split()
    features['unique_words'] = len(set(words))
    features['unique_ratio'] = len(set(words)) / len(words) if words else 0
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    
    features['has_graph'] = int(any(w in text for w in 
        ['graph', 'tree', 'node', 'edge', 'vertex', 'path', 'cycle', 'dag', 'directed', 'undirected']))
    features['has_dp'] = int(any(w in text for w in 
        ['dynamic programming', 'dp', 'memoization', 'optimal substructure', 'overlapping subproblem']))
    features['has_greedy'] = int('greedy' in text)
    features['has_binary_search'] = int(any(w in text for w in 
        ['binary search', 'bisection', 'lower bound', 'upper bound']))
    features['has_sorting'] = int(any(w in text for w in ['sort', 'sorted', 'sorting']))
    features['has_dfs_bfs'] = int(any(w in text for w in 
        ['dfs', 'bfs', 'depth first', 'breadth first', 'traversal']))
    features['has_shortest_path'] = int(any(w in text for w in 
        ['shortest path', 'dijkstra', 'bellman', 'floyd']))
    features['has_advanced_ds'] = int(any(w in text for w in 
        ['segment tree', 'fenwick', 'trie', 'suffix array', 'union find', 'disjoint set']))
    features['has_flow'] = int(any(w in text for w in 
        ['max flow', 'min cut', 'network flow', 'bipartite matching']))
    features['has_string_algo'] = int(any(w in text for w in 
        ['substring', 'palindrome', 'kmp', 'lcs', 'edit distance', 'pattern']))

    features['has_number_theory'] = int(any(w in text for w in 
        ['modulo', 'prime', 'gcd', 'lcm', 'factorial', 'coprime', 'euler']))
    features['has_combinatorics'] = int(any(w in text for w in 
        ['permutation', 'combination', 'binomial', 'catalan']))
    features['has_probability'] = int(any(w in text for w in 
        ['probability', 'expected value', 'random']))
    features['has_geometry'] = int(any(w in text for w in 
        ['geometry', 'coordinate', 'polygon', 'convex hull', 'point', 'line', 'angle']))
    features['has_matrix'] = int('matrix' in text or 'matrices' in text)

    features['has_optimization'] = int(any(w in text for w in 
        ['minimum', 'maximum', 'minimize', 'maximize', 'optimal', 'best']))
    features['count_minimum'] = text.count('minimum')
    features['count_maximum'] = text.count('maximum')

    numbers = re.findall(r'\d+', text)
    if numbers:
        nums = [int(n) for n in numbers if len(n) <= 10]
        if nums:
            max_num = max(nums)
            features['max_constraint'] = min(max_num, 1e9)
            features['min_constraint'] = min(nums)
            features['avg_constraint'] = np.mean(nums)
            features['log_max_constraint'] = np.log10(max_num + 1)
            features['num_constraints'] = len(nums)
            features['tiny_constraint'] = int(max_num <= 20)
            features['small_constraint'] = int(max_num <= 100)
            features['medium_constraint'] = int(max_num <= 1000)
            features['large_constraint'] = int(1000 < max_num <= 100000)
            features['huge_constraint'] = int(max_num > 100000)
        else:
            for key in ['max_constraint', 'min_constraint', 'avg_constraint', 'log_max_constraint', 
                       'num_constraints', 'tiny_constraint', 'small_constraint', 
                       'medium_constraint', 'large_constraint', 'huge_constraint']:
                features[key] = 0
    else:
        for key in ['max_constraint', 'min_constraint', 'avg_constraint', 'log_max_constraint',
                   'num_constraints', 'tiny_constraint', 'small_constraint',
                   'medium_constraint', 'large_constraint', 'huge_constraint']:
            features[key] = 0

    features['modulo_count'] = text.count('modulo') + text.count(' mod ')
    features['has_queries'] = int('quer' in text)
    features['query_count'] = text.count('query') + text.count('queries')
    features['testcase_count'] = text.count('test case') + text.count('testcase')
    features['formula_count'] = text.count('$')
    
    sentences = [s for s in text.split('.') if s.strip()]
    features['num_sentences'] = len(sentences)
    features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
 
    complexity_score = (
        features['has_dp'] * 3 +
        features['has_graph'] * 2 +
        features['has_advanced_ds'] * 4 +
        features['has_flow'] * 5 +
        features['has_shortest_path'] * 2 +
        features['has_number_theory'] * 2 +
        features['huge_constraint'] * 3 +
        features['has_string_algo'] * 2
    )
    features['algo_complexity_score'] = complexity_score
    
    return features

def calculate_text_stats(problem_text, input_text, output_text):
    """Calculate statistics for display in UI"""
    full_text = problem_text + " " + input_text + " " + output_text
    words = full_text.split()
    word_count = len(words)
    char_count = len(full_text)
    
    keywords = {
        'Graph/Tree': ['graph', 'tree', 'node', 'edge'],
        'Dynamic Programming': ['dynamic', 'dp', 'memoization'],
        'Greedy': ['greedy'],
        'Sorting': ['sort', 'sorted'],
        'Binary Search': ['binary search', 'bisection'],
        'String Algorithms': ['substring', 'palindrome', 'pattern'],
        'Math': ['modulo', 'prime', 'gcd'],
        'Optimization': ['minimum', 'maximum', 'optimize']
    }
    
    found_categories = []
    text_lower = full_text.lower()
    
    for category, terms in keywords.items():
        if any(term in text_lower for term in terms):
            found_categories.append(category)
    
    numbers = re.findall(r'\d+', text_lower)
    max_constraint = 0
    if numbers:
        nums = [int(n) for n in numbers if len(n) <= 10]
        if nums:
            max_constraint = max(nums)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'complexity_indicators': len(found_categories),
        'found_keywords': found_categories,
        'max_constraint': max_constraint
    }

def get_prediction_details(classifier, regressor, text_vector, numerical_features, scaler):
    """
    Get predictions from both classifier and regressor
    
    Args:
        classifier: trained classification model
        regressor: trained regression model
        text_vector: vectorized text features
        numerical_features: dict of numerical features
        scaler: fitted StandardScaler
    
    Returns:
        predicted_class, predicted_score, confidence, prob_dict
    """
    # Build feature vector aligned with scaler expectations
    provided_vals = list(numerical_features.values())

    if hasattr(scaler, 'feature_names_in_'):
        feature_names = list(scaler.feature_names_in_)
        feature_values = np.array([[numerical_features.get(n, 0.0) for n in feature_names]])
    else:
        # Fallback: determine expected feature count from scaler's learned attributes
        try:
            expected_n = int(getattr(scaler, 'n_features_in_', getattr(scaler, 'mean_', None).shape[0]))
        except Exception:
            expected_n = len(provided_vals)

        if len(provided_vals) == expected_n:
            feature_values = np.array([provided_vals])
        elif len(provided_vals) < expected_n:
            padded = provided_vals + [0.0] * (expected_n - len(provided_vals))
            feature_values = np.array([padded])
        else:
            # More provided than expected: truncate and warn
            feature_values = np.array([provided_vals[:expected_n]])

    numerical_scaled = scaler.transform(feature_values)
    
    combined_features = hstack([csr_matrix(numerical_scaled), text_vector])

    predicted_class = classifier.predict(combined_features)[0]
    predicted_score = regressor.predict(combined_features)[0]

    if hasattr(classifier, 'predict_proba'):
        class_probabilities = classifier.predict_proba(combined_features)[0]
        class_labels = classifier.classes_
        max_confidence = max(class_probabilities) * 100
        prob_dict = {class_labels[i]: class_probabilities[i] * 100 
                    for i in range(len(class_labels))}
    else:
        max_confidence = 85.0
        prob_dict = {predicted_class: max_confidence}
    
    return predicted_class, predicted_score, max_confidence, prob_dict

def find_most_similar(user_text, vectorizer, scaler, problems_df, top_k=5):
    """Find similar problems from dataset"""
    if problems_df is None or problems_df.empty:
        return []
    
    user_cleaned = clean_text(user_text)
    if not user_cleaned:
        return []
    
    try:
        user_vector = vectorizer.transform([user_cleaned])
        
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
        
        similarity_scores = cosine_similarity(user_vector, problem_vectors)[0]
        
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
        print(f"Error in find_most_similar: {e}")
        return []

def format_difficulty_display(difficulty_class):
    """Get emoji for difficulty class"""
    difficulty_lower = difficulty_class.lower()
    emoji_map = {
        "easy": "🟢",
        "medium": "🟡",
        "hard": "🔴"
    }
    return emoji_map.get(difficulty_lower, "⚪")

def get_difficulty_description(difficulty_class):
    """Get description for difficulty level"""
    descriptions = {
        "easy": "Suitable for beginners. Basic algorithms and data structures.",
        "medium": "Requires good understanding of algorithms and problem-solving skills.",
        "hard": "Advanced algorithms, complex data structures, or mathematical insights required."
    }
    return descriptions.get(difficulty_class.lower(), "Unknown difficulty level")
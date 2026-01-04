import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import utils

# Page config
st.set_page_config(page_title="AutoJudge", page_icon="⚖️", layout="wide")

# Load models once
@st.cache_resource
def load_models():
    text_vectorizer = joblib.load("models/text_vectorizer.pkl")
    classifier = joblib.load("models/difficulty_classifier.pkl")
    regressor = joblib.load("models/score_regressor.pkl")
    return text_vectorizer, classifier, regressor

vectorizer, difficulty_classifier, score_regressor = load_models()

# Load dataset for similar problems
@st.cache_data
def load_problems_data():
    try:
        problems_df = pd.read_csv("data/problems.csv")
        return problems_df
    except:
        return None

problems_data = load_problems_data()

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Main UI
st.title("⚖️ AutoJudge - Problem Difficulty Analyzer")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Navigation")
    page = st.radio("Choose a page", ["Single Prediction", "Batch Prediction", "Prediction History", "About"])
    st.markdown("---")
    
    if page == "Prediction History" and st.session_state.prediction_history:
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

# Single Prediction Page
if page == "Single Prediction":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Problem Details")
        problem_desc = st.text_area("Problem Description", height=150, 
                                    placeholder="Enter the problem statement here...")
        input_desc = st.text_area("Input Description", height=100,
                                 placeholder="Describe the input format...")
        output_desc = st.text_area("Output Description", height=100,
                                  placeholder="Describe the expected output...")
    
    with col2:
        st.subheader("📈 Quick Stats")
        if problem_desc or input_desc or output_desc:
            stats = utils.calculate_text_stats(problem_desc, input_desc, output_desc)
            st.metric("Word Count", stats['word_count'])
            st.metric("Characters", stats['char_count'])
            st.metric("Complexity Indicators", stats['complexity_indicators'])
            if stats['found_keywords']:
                st.write("**Keywords Found:**")
                for keyword in stats['found_keywords']:
                    st.write(f"• {keyword}")
    
    st.markdown("---")
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        predict_btn = st.button("🔮 Predict Difficulty", type="primary", use_container_width=True)
    
    if predict_btn:
        if not problem_desc.strip():
            st.warning("⚠️ Please enter at least a problem description.")
        else:
            with st.spinner("Analyzing problem..."):
                combined_text = problem_desc + " " + input_desc + " " + output_desc
                cleaned_text = utils.clean_text(combined_text)
                
                if not cleaned_text:
                    st.error("❌ Unable to process the input text.")
                else:
                    text_vector = vectorizer.transform([cleaned_text])
                    
                    predicted_class, predicted_score, confidence, prob_dict = utils.get_prediction_details(
                        difficulty_classifier, score_regressor, text_vector
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    # Color coding
                    emoji = utils.format_difficulty_display(predicted_class)
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("Difficulty Class", f"{emoji} {predicted_class.upper()}")
                    
                    with result_col2:
                        st.metric("Difficulty Score", f"{predicted_score:.2f}")
                    
                    with result_col3:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Probability breakdown
                    st.markdown("### 📊 Class Probabilities")
                    prob_df = pd.DataFrame(list(prob_dict.items()), columns=['Class', 'Probability (%)'])
                    prob_df = prob_df.sort_values('Probability (%)', ascending=False)
                    st.bar_chart(prob_df.set_index('Class'))
                    
                    # Save to history
                    prediction_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'problem_desc': problem_desc[:100] + "..." if len(problem_desc) > 100 else problem_desc,
                        'predicted_class': predicted_class,
                        'predicted_score': float(predicted_score),
                        'confidence': float(confidence)
                    }
                    st.session_state.prediction_history.append(prediction_entry)
                    
                    # Similar problems
                    st.markdown("---")
                    st.subheader("🔍 Similar Problems")
                    similar = utils.find_most_similar(combined_text, vectorizer, problems_data, top_k=3)
                    
                    if similar:
                        for sim_prob in similar:
                            with st.expander(f"📌 {sim_prob['title']} (Similarity: {sim_prob['similarity']:.2%})"):
                                st.write(f"**Class:** {sim_prob['class'].upper()}")
                                st.write(f"**Score:** {sim_prob['score']}")
                                if sim_prob['url']:
                                    st.write(f"**Link:** {sim_prob['url']}")
                    else:
                        st.info("No similar problems found in the dataset.")

# Batch Prediction Page
elif page == "Batch Prediction":
    st.subheader("📦 Batch Prediction")
    st.write("Upload a CSV file with columns: 'description', 'input_description', 'output_description'")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_cols = ['description', 'input_description', 'output_description']
            
            if all(col in batch_df.columns for col in required_cols):
                st.success(f"✅ Loaded {len(batch_df)} problems")
                st.dataframe(batch_df.head())
                
                if st.button("🚀 Predict All", type="primary"):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in batch_df.iterrows():
                        combined = str(row['description']) + " " + str(row['input_description']) + " " + str(row['output_description'])
                        cleaned = utils.clean_text(combined)
                        
                        if cleaned:
                            text_vec = vectorizer.transform([cleaned])
                            pred_class, pred_score, conf, _ = utils.get_prediction_details(
                                difficulty_classifier, score_regressor, text_vec
                            )
                            
                            results.append({
                                'title': row.get('title', f'Problem {idx+1}'),
                                'predicted_class': pred_class,
                                'predicted_score': round(pred_score, 2),
                                'confidence': round(conf, 1)
                            })
                        
                        progress_bar.progress((idx + 1) / len(batch_df))
                    
                    results_df = pd.DataFrame(results)
                    st.subheader("📊 Batch Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button("💾 Download Results", csv, "batch_predictions.csv", "text/csv")
            else:
                st.error("❌ CSV must contain columns: description, input_description, output_description")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

# Prediction History Page
elif page == "Prediction History":
    st.subheader("📜 Prediction History")
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Statistics
        st.markdown("### 📈 History Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            class_counts = history_df['predicted_class'].value_counts()
            st.write("**Class Distribution:**")
            for cls, count in class_counts.items():
                st.write(f"{cls}: {count}")
        
        with col2:
            avg_score = history_df['predicted_score'].mean()
            st.metric("Average Score", f"{avg_score:.2f}")
        
        with col3:
            avg_conf = history_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_conf:.1f}%")
        
        # Export
        if st.button("💾 Export History"):
            csv = history_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "prediction_history.csv", "text/csv")
    else:
        st.info("No prediction history yet. Make some predictions first!")

# About Page
elif page == "About":
    st.subheader("ℹ️ About AutoJudge")
    st.write("""
    **AutoJudge** is an intelligent system that automatically predicts the difficulty of programming problems
    based solely on their textual descriptions.
    
    ### Features:
    - 🎯 **Single Problem Prediction**: Get instant difficulty predictions with confidence scores
    - 📦 **Batch Processing**: Analyze multiple problems at once
    - 📊 **Problem Statistics**: View word count, complexity indicators, and more
    - 🔍 **Similar Problems**: Find problems similar to your input
    - 📜 **Prediction History**: Track all your predictions
    
    ### How it works:
    The system uses machine learning models trained on a dataset of programming problems to predict:
    - **Difficulty Class**: Easy, Medium, or Hard
    - **Difficulty Score**: A numerical score indicating problem complexity
    
    All predictions are based on analyzing the problem description, input format, and output requirements.
    """)
    
    st.markdown("---")
    st.write("**Built with:** Streamlit, scikit-learn, pandas, numpy")

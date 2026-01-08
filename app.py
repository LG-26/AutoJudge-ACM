import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import utils
from scipy.sparse import hstack

st.set_page_config(
    page_title="AutoJudge - Problem Difficulty Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        try:
            vec_word = joblib.load("models/text_vectorizer_word.pkl")
            vec_char = joblib.load("models/text_vectorizer_char.pkl")
            class CombinedVectorizer:
                def __init__(self, w, c):
                    self.w = w
                    self.c = c
                def transform(self, texts):
                    wv = self.w.transform(texts)
                    cv = self.c.transform(texts)
                    return hstack([wv, cv])

            vectorizer = CombinedVectorizer(vec_word, vec_char)
        except Exception:
            vectorizer = joblib.load("models/text_vectorizer.pkl")

        scaler = joblib.load("models/numerical_scaler.pkl")
        classifier = joblib.load("models/difficulty_classifier.pkl")
        regressor = joblib.load("models/score_regressor.pkl")
        metadata = joblib.load("models/metadata.pkl")
        return vectorizer, scaler, classifier, regressor, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the training notebook first to generate models.")
        st.stop()

vectorizer, scaler, classifier, regressor, metadata = load_models()

@st.cache_data
def load_problems_data():
    """Load problems dataset for similarity search"""
    try:
        return pd.read_csv("data/problems.csv")
    except:
        return None

problems_data = load_problems_data()

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

st.markdown('<p class="main-header">⚖️ AutoJudge</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>Intelligent Problem Difficulty Analyzer</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("📊 Model Information")
    
    if metadata:
        st.metric("Classification Accuracy", f"{metadata.get('classifier_accuracy', 0)*100:.1f}%")
        st.metric("Regression MAE", f"{metadata.get('regressor_mae', 0):.2f}")
        st.caption(f"Model: {metadata.get('classifier_type', 'Unknown')}")
    
    st.markdown("---")
    st.header("🔍 Navigation")
    page = st.radio("", ["🎯 Predict", "📦 Batch Process", "📜 History", "ℹ️ About"], label_visibility="collapsed")
    
    st.markdown("---")
    
    if page == "📜 History" and st.session_state.prediction_history:
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

if page == "🎯 Predict":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Problem Details")
        
        problem_desc = st.text_area(
            "Problem Description *",
            height=200,
            placeholder="Enter the main problem statement here...\n\nExample: Given an array of integers, find two numbers that add up to a target value.",
            help="Provide the complete problem description"
        )
        
        col_input, col_output = st.columns(2)
        
        with col_input:
            input_desc = st.text_area(
                "Input Format",
                height=120,
                placeholder="Describe input format...\n\nExample: First line: integer N\nSecond line: N integers",
                help="Describe the input format and constraints"
            )
        
        with col_output:
            output_desc = st.text_area(
                "Output Format",
                height=120,
                placeholder="Describe expected output...\n\nExample: Two integers separated by space",
                help="Describe the expected output format"
            )
    
    with col2:
        st.subheader("📈 Text Analysis")
        
        if problem_desc or input_desc or output_desc:
            stats = utils.calculate_text_stats(problem_desc, input_desc, output_desc)
            
            st.metric("📊 Word Count", stats['word_count'])
            st.metric("📏 Characters", stats['char_count'])
            st.metric("🧩 Complexity Score", stats['complexity_indicators'])
            
            if stats['max_constraint'] > 0:
                st.metric("📐 Max Constraint", f"{stats['max_constraint']:,}")
            
            if stats['found_keywords']:
                st.markdown("**🔑 Detected Concepts:**")
                for keyword in stats['found_keywords'][:5]:
                    st.markdown(f"• {keyword}")
        else:
            st.info("Enter problem details to see analysis")
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("🔮 Analyze Problem", type="primary", use_container_width=True)
    
    if predict_btn:
        if not problem_desc.strip():
            st.warning("⚠️ Please enter at least a problem description.")
        else:
            with st.spinner("🤖 Analyzing problem..."):
                combined_text = problem_desc + " " + input_desc + " " + output_desc
                cleaned_text = utils.clean_text(combined_text)
                
                if not cleaned_text:
                    st.error("❌ Unable to process the input text.")
                else:
                    numerical_features = utils.extract_features_from_text(
                        problem_desc, input_desc, output_desc
                    )
                    
                    text_vector = vectorizer.transform([cleaned_text])
                    
                    predicted_class, predicted_score, confidence, prob_dict = utils.get_prediction_details(
                        classifier, regressor, text_vector, numerical_features, scaler
                    )
                    
                    st.markdown("---")
                    st.subheader("🎯 Analysis Results")
                    
                    emoji = utils.format_difficulty_display(predicted_class)
                    description = utils.get_difficulty_description(predicted_class)
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="text-align: center; font-size: 3rem;">{emoji}</h3>
                            <h2 style="text-align: center; color: #667eea;">{predicted_class.upper()}</h2>
                            <p style="text-align: center; color: #666;">{description}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #666;">Difficulty Score</h4>
                            <h1 style="color: #764ba2; font-size: 3rem;">{predicted_score:.2f}</h1>
                            <p style="color: #666;">Out of 10.0</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with result_col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #666;">Confidence</h4>
                            <h1 style="color: #667eea; font-size: 3rem;">{confidence:.0f}%</h1>
                            <p style="color: #666;">Model certainty</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 📊 Class Probabilities")
                    prob_df = pd.DataFrame([
                        {"Class": k.upper(), "Probability": v} 
                        for k, v in prob_dict.items()
                    ]).sort_values('Probability', ascending=False)
                    
                    st.bar_chart(prob_df.set_index('Class')['Probability'])
                    
                    prediction_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'problem_desc': problem_desc[:100] + "..." if len(problem_desc) > 100 else problem_desc,
                        'predicted_class': predicted_class,
                        'predicted_score': float(predicted_score),
                        'confidence': float(confidence)
                    }
                    st.session_state.prediction_history.append(prediction_entry)
                    
                    if problems_data is not None:
                        st.markdown("---")
                        st.subheader("🔍 Similar Problems")
                        
                        similar = utils.find_most_similar(
                            combined_text, vectorizer, scaler, problems_data, top_k=3
                        )
                        
                        if similar:
                            for i, sim_prob in enumerate(similar):
                                with st.expander(f"📌 {sim_prob['title']} (Similarity: {sim_prob['similarity']:.1%})"):
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.write(f"**Class:** {sim_prob['class'].upper()}")
                                        st.write(f"**Score:** {sim_prob['score']}")
                                    with col_b:
                                        if sim_prob['url']:
                                            st.markdown(f"[View Problem]({sim_prob['url']})")
                        else:
                            st.info("No similar problems found in dataset")

elif page == "📦 Batch Process":
    st.subheader("📦 Batch Problem Analysis")
    st.write("Upload a CSV file to analyze multiple problems at once")
    
    with st.expander("📋 See CSV Format"):
        st.code("""title,description,input_description,output_description
"Two Sum","Given array find two numbers that sum to target","Line 1: N, Line 2: N integers, Line 3: target","Two space-separated integers"
"Binary Search","Search for element in sorted array","Line 1: N, Line 2: N sorted integers, Line 3: target","Index of target or -1"
        """, language="csv")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_cols = ['description']
            
            if 'description' in batch_df.columns:
                st.success(f"✅ Loaded {len(batch_df)} problems")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if 'input_description' not in batch_df.columns:
                    batch_df['input_description'] = ''
                if 'output_description' not in batch_df.columns:
                    batch_df['output_description'] = ''
                
                if st.button("🚀 Analyze All Problems", type="primary"):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in batch_df.iterrows():
                        status_text.text(f"Processing problem {idx+1}/{len(batch_df)}...")
                        
                        combined = (str(row.get('description', '')) + " " + 
                                  str(row.get('input_description', '')) + " " + 
                                  str(row.get('output_description', '')))
                        cleaned = utils.clean_text(combined)
                        
                        if cleaned:
                            numerical_features = utils.extract_features_from_text(
                                str(row.get('description', '')),
                                str(row.get('input_description', '')),
                                str(row.get('output_description', ''))
                            )
                            
                            text_vec = vectorizer.transform([cleaned])
                            pred_class, pred_score, conf, _ = utils.get_prediction_details(
                                classifier, regressor, text_vec, numerical_features, scaler
                            )
                            
                            results.append({
                                'title': row.get('title', f'Problem {idx+1}'),
                                'predicted_class': pred_class.upper(),
                                'predicted_score': round(pred_score, 2),
                                'confidence': round(conf, 1)
                            })
                        
                        progress_bar.progress((idx + 1) / len(batch_df))
                    
                    status_text.text("✅ Analysis complete!")

                    results_df = pd.DataFrame(results)
                    st.subheader("📊 Batch Results")
                    st.dataframe(results_df, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Problems", len(results_df))
                    with col2:
                        avg_score = results_df['predicted_score'].mean()
                        st.metric("Average Score", f"{avg_score:.2f}")
                    with col3:
                        mode_class = results_df['predicted_class'].mode()[0]
                        st.metric("Most Common", mode_class)

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "💾 Download Results",
                        csv,
                        "batch_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.error("❌ CSV must contain at least a 'description' column")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")


elif page == "📜 History":
    st.subheader("📜 Prediction History")
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        st.dataframe(history_df, use_container_width=True)
        
        st.markdown("### 📈 Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        
        with col2:
            avg_score = history_df['predicted_score'].mean()
            st.metric("Avg Score", f"{avg_score:.2f}")
        
        with col3:
            avg_conf = history_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        with col4:
            mode_class = history_df['predicted_class'].mode()[0]
            st.metric("Most Common", mode_class.upper())

        st.markdown("### 📊 Class Distribution")
        class_counts = history_df['predicted_class'].value_counts()
        st.bar_chart(class_counts)

        csv = history_df.to_csv(index=False)
        st.download_button(
            "💾 Export History",
            csv,
            "prediction_history.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("📭 No prediction history yet. Make some predictions first!")


elif page == "ℹ️ About":
    st.subheader("ℹ️ About AutoJudge")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What is AutoJudge?
        
        **AutoJudge** is an intelligent system that automatically predicts the difficulty 
        of competitive programming problems using machine learning.
        
        ### ✨ Features
        
        - 🎯 **Accurate Predictions**: Advanced ML models with 70%+ accuracy
        - 📊 **Dual Output**: Both difficulty class and numerical score
        - 🔍 **Similarity Search**: Find similar problems in the database
        - 📦 **Batch Processing**: Analyze multiple problems at once
        - 📜 **History Tracking**: Keep track of all predictions
        - 🧩 **Smart Analysis**: Detects algorithms, constraints, and complexity
        
        ### 🤖 How It Works
        
        The system analyzes:
        - Problem description text
        - Input/output format specifications
        - Algorithmic concepts mentioned
        - Constraint sizes and complexity hints
        - Mathematical indicators
        """)
    
    with col2:
        st.markdown(f"""
        ### 📊 Model Performance
        
        **Classification Model:** {metadata.get('classifier_type', 'Unknown')}
        - Accuracy: **{metadata.get('classifier_accuracy', 0)*100:.1f}%**
        - F1 Score: **{metadata.get('classifier_f1', 0):.3f}**
        
        **Regression Model:** Random Forest
        - MAE: **{metadata.get('regressor_mae', 0):.2f}**
        - RMSE: **{metadata.get('regressor_rmse', 0):.2f}**
        
        ### 🔧 Technical Stack
        
        - **Machine Learning**: scikit-learn, XGBoost
        - **NLP**: TF-IDF Vectorization
        - **Features**: {metadata.get('n_total_features', 'N/A')} total
        - **UI**: Streamlit
        - **Data Processing**: Pandas, NumPy
        
        ### 📖 Difficulty Levels
        
        🟢 **Easy**: Basic algorithms, simple logic
        🟡 **Medium**: Intermediate algorithms, data structures
        🔴 **Hard**: Advanced algorithms, complex optimizations
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ for competitive programmers</p>
        <p>Powered by Machine Learning & Natural Language Processing</p>
    </div>
    """, unsafe_allow_html=True)
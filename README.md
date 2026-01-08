# AutoJudge

## Project Overview
AutoJudge is a small ML-powered tool that analyzes competitive programming problem text and predicts a difficulty class (easy / medium / hard) and a numerical difficulty score (0–10). It also provides text analysis, similarity search against a problems dataset, and batch processing via a CSV upload.

## Dataset Used
The project uses a problems dataset stored at `data/problems.csv`. The dataset contains problem metadata such as `title`, `description`, `input_description`, `output_description`, `problem_score`, `problem_class`, and `url` (if available). Training was performed on a curated dataset of competitive programming problems and their labeled difficulty/score.

## Approach and Models Used
- Text features: dual TF-IDF vectorizers (word-level and char-level) saved as `models/text_vectorizer_word.pkl` and `models/text_vectorizer_char.pkl` which are combined at inference time.
- Numerical features: hand-engineered features extracted from the problem description and input/output text (see `utils.extract_features_from_text`). These features are scaled with `models/numerical_scaler.pkl`.
- Classifier: difficulty class model saved as `models/difficulty_classifier.pkl` (e.g., XGBoost / RandomForest / scikit-learn estimator).
- Regressor: numerical difficulty score model saved as `models/score_regressor.pkl` (e.g., RandomForest regressor).
- Metadata: saved training metrics and configuration in `models/metadata.pkl`.

## Evaluation Metrics
- Classification: Accuracy (reported in `metadata['classifier_accuracy']`) and F1 score (`metadata['classifier_f1']`).
- Regression: Mean Absolute Error (MAE, `metadata['regressor_mae']`) and Root Mean Squared Error (RMSE, `metadata['regressor_rmse']`).

## Steps to Run Locally
1. Create a Python environment (recommended: venv or conda).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the `models/` directory contains the trained artifacts (`text_vectorizer_word.pkl`, `text_vectorizer_char.pkl`, `numerical_scaler.pkl`, `difficulty_classifier.pkl`, `score_regressor.pkl`, `metadata.pkl`). If your training produced a single `text_vectorizer.pkl`, the app will fall back to that.

4. Start the Streamlit app from the project root:

```bash
streamlit run app.py
```

5. Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Web Interface Explanation
The Streamlit UI (in `app.py`) contains the following pages (select from the sidebar):
- `🎯 Predict`: Enter a problem description, input/output formats, then click **Analyze Problem** to get predicted difficulty class, numerical score, confidence, and class probabilities. The app also shows detected keywords and text statistics.
- `📦 Batch Process`: Upload a CSV with at least a `description` column (optionally `title`, `input_description`, `output_description`) to analyze multiple problems at once. Results can be downloaded as CSV.
- `📜 History`: View recent predictions made in the current session and export history.
- `ℹ️ About`: Project details and model performance metrics from `models/metadata.pkl`.

## Demo Video
Add a 2–3 minute demo video link here:
- Demo: https://example.com/demo (replace with your actual video URL)

## Author / Contact
- Name: [Your Name Here]
- Email / Profile: [Add your contact details or GitHub link]

---

If you want, I can:
- run a quick test to load models and perform a sample prediction, or
- start the Streamlit app here and report any runtime errors.

Replace placeholders (demo URL and author details) with your real information before sharing the repo.
# src/traditional_ai/train.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import utils  # Assuming utils.py is in the same directory or Python path is set up

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "twcs.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed" / "traditional_ai"
MODEL_DIR = BASE_DIR / "models" / "traditional_ai"

# Ensure output directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib" # chose this model

# --- Helper Functions ---
def prepare_training_data(df_full, target_companies):
    """Prepares training data by filtering and cleaning."""
    inbound_df = df_full[df_full['inbound']].copy()
    results = inbound_df['text'].apply(
        lambda t: pd.Series(utils.extract_company_and_text(t, target_companies))
    )
    inbound_df[['extracted_company_label', 'text_for_feature']] = results
    inbound_df.dropna(subset=['extracted_company_label'], inplace=True)
    inbound_df['cleaned_text_for_feature'] = inbound_df['text_for_feature'].apply(utils.general_text_cleaning)
    return inbound_df

def save_label_mappings(label_encoder, output_path):
    """Saves label mappings to a CSV file."""
    label_mapping_df = pd.DataFrame({
        'label_name': label_encoder.classes_,
        'encoded_label': range(len(label_encoder.classes_))
    })
    label_mapping_df.to_csv(output_path, index=False)
    print(f"Label mappings saved to {output_path}")

def save_artifacts(tfidf_vectorizer, label_encoder, model):
    """Saves model artifacts to disk."""
    joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"Artifacts saved: TF-IDF Vectorizer -> {TFIDF_VECTORIZER_PATH}, "
          f"Label Encoder -> {LABEL_ENCODER_PATH}, Model -> {MODEL_PATH}")

# --- Main Training Logic ---
def train_model():
    print("Starting model training process...")

    # 1. Load Data
    print(f"Loading data from {RAW_DATA_PATH}...")
    df_full = utils.load_data(RAW_DATA_PATH)

    # 2. Identify Target Companies
    print("Identifying target companies...")
    target_companies = utils.get_target_companies(df_full, top_n=10)
    if not target_companies:
        print("No target companies identified. Exiting.")
        return
    print(f"Target companies: {target_companies}")

    # 3. Prepare Training Data
    print("Preparing training data...")
    inbound_df = prepare_training_data(df_full, target_companies)
    if inbound_df.empty:
        print("No training data available after filtering. Exiting.")
        return

    X = inbound_df['cleaned_text_for_feature']
    y_raw_labels = inbound_df['extracted_company_label']

    # 4. Encode Labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw_labels)
    save_label_mappings(label_encoder, PROCESSED_DATA_DIR / "label_mappings.csv")

    # 5. Split Data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. TF-IDF Vectorization
    print("Performing TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # 7. Train Model
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # 8. Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # 9. Save Artifacts
    print("Saving artifacts...")
    save_artifacts(tfidf_vectorizer, label_encoder, model)

    print("\nTraining process completed.")

if __name__ == '__main__':
    train_model()
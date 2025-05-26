# src/traditional_ai/predict.py
import joblib
from pathlib import Path
import utils # From src/traditional_ai/utils.py

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models" / "traditional_ai"

TFIDF_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib"

# load saved artifacts
tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
model = joblib.load(MODEL_PATH)
# load classes
TARGET_COMPANIES_FROM_TRAINING = list(label_encoder.classes_)

def predict(raw_text: str):

    #dont need the label for predict
    _, text_for_feature = utils.extract_company_and_clean_text(raw_text, TARGET_COMPANIES_FROM_TRAINING)
    #like in train
    cleaned_text_for_feature = utils.general_text_cleaning(text_for_feature)
    text_tfidf = tfidf_vectorizer.transform(cleaned_text_for_feature)
    
    # find the label
    predicted_label_encoded = model.predict(text_tfidf)
    #company's name
    predicted_label_name = label_encoder.inverse_transform(predicted_label_encoded)[0]
    return predicted_label_name

if __name__ == '__main__':
    sample_texts = [
    "My new iPhone screen is flickering, can @AppleSupport help?",
    "Having trouble with my @AmazonHelp order, it's been delayed for a week!",
    "Hey @SpotifyCares, my playlists are not syncing across devices.",
    "Just a random tweet about how much I love coffee.", # Should ideally not map or map to a default if we had one
    "@UnknownCompany my product broke!" # This company was likely not in training
    ]

    for raw_text in sample_texts:
        prediction = predict(raw_text)
    print(f"Input: \"{raw_text}\"")
    if prediction:
        print(f"Predicted Company: @{prediction}\n") # Assuming labels are stored without @
    else:
        print("Predicted Company: None (or input not suitable for model)\n")
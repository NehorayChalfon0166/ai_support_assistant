# src/traditional_ai/utils.py
import pandas as pd
import re
from collections import Counter
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK is available
try:
    stopwords.words('english')
    word_tokenize("test")
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

def load_data(file_path):
    """Loads the TWCS dataset."""
    return pd.read_csv(file_path)

def get_target_companies(df, top_n=10):
    """
    Identifies target companies.
    1. From author_id of outbound (company) tweets.
    2. From most frequent @mentions in inbound (user) tweets that are actual companies.
    """
    outbound_df = df[df['inbound'] == False]
    company_author_ids = set(outbound_df['author_id'].unique())

    inbound_df = df[df['inbound'] == True]
    mentions = []
    for text in inbound_df['text']:
        mentions.extend(re.findall(r'@\w+', text))
    
    cleaned_mentions = [mention.lstrip('@') for mention in mentions]
    
    # Filter mentions to include only those that are actual company author_ids
    valid_company_mentions = [m for m in cleaned_mentions if m in company_author_ids]
    
    mention_counts = Counter(valid_company_mentions)
    
    target_companies = [mention for mention, _ in mention_counts.most_common(top_n)]
    return target_companies

def extract_company_and_clean_text(text, target_companies):
    """
    Extracts the first valid company @-mention from the text and
    returns the company label and the text with the @-mention removed.
    """
    mentions = re.findall(r'@\w+', text)
    extracted_label = None
    cleaned_text = text

    for mention in mentions:
        mention_cleaned = mention.lstrip('@')
        if mention_cleaned in target_companies:
            extracted_label = mention_cleaned
            # Remove only the first occurrence of the specific @mention
            cleaned_text = re.sub(r'@' + re.escape(extracted_label), '', text, 1).strip()
            break # Found the first target company
            
    return extracted_label, cleaned_text

def general_text_cleaning(text):
    """Applies general cleaning: lowercase, punctuation, stopwords."""
    text = str(text).lower() # Ensure text is string
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words_list = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop_words_list and word.isalpha()]
    return ' '.join(tokens)
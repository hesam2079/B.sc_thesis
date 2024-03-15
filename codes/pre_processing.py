import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




# Function to remove URLs from text
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

# Function to remove mentions from text
def remove_mentions(text):
    mention_pattern = re.compile(r'@\w+')
    return mention_pattern.sub('', text)

# Function to cleaning the text
def clean_text(text):
    # Remove special characters, hashtags, and mentions
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    #cleaned_text = re.sub(r"#\w+", "", cleaned_text)
    cleaned_text = re.sub(r"@\w+", "", cleaned_text)

    # Convert text to lowercase
    cleaned_text = cleaned_text.lower()

    # Remove extra whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text

# Function to tokenize
def tokenize_text(text):
    # Tokenize the text into individual words/tokens
    tokens = word_tokenize(text)
    return tokens

# Function to remove stop words from text
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

# Function to remove punctuations and numbers
def remove_punctuations_and_numbers(text):
    # Remove punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    return text

# Lemmatizer Function
def perform_lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Stemmer Function
def perform_stemming(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def join_tokens(tokens):
    joined_text = ' '.join(tokens)
    return joined_text

# Load the dataset into a pandas DataFrame
df = pd.read_csv('tweet_emotions.csv')

# Apply def's
df['text'] = df['text'].apply(remove_urls)
df['text'] = df['text'].apply(remove_mentions)
df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].apply(remove_punctuations_and_numbers)
df['tokenized'] = df['text'].apply(tokenize_text)
#df['tokenized'] = df['tokenized'].apply(remove_stopwords)
df['lemmatized'] = df['tokenized'].apply(perform_lemmatization)
df['stemmed'] = df['tokenized'].apply(perform_stemming)
df['joined_stemme'] = df['stemmed'].apply(join_tokens)
df['joined_lemmatized'] = df['lemmatized'].apply(join_tokens)

# Display the updated DataFrame
print(df.head())

# Save to pre_proccessed.csv
df.to_csv('pre_proccessed.csv', index=False)
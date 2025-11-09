import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

def clean_text(text):
    """
    Comprehensive text cleaning function for NLP sentiment classification.
    
    Steps performed:
    1. Convert to lowercase for consistency
    2. Remove URLs (http/https links)
    3. Remove mentions (@username)
    4. Remove hashtag symbols but keep the text (e.g., #disaster -> disaster)
    5. Remove HTML entities and special characters
    6. Remove punctuation
    8. Remove extra whitespaces
    9. Remove stopwords (common words like 'the', 'is', 'at')
    10. Lemmatization (convert words to their base form)
    
    Args:
        text (str): Raw text to be cleaned
    
    Returns:
        str: Cleaned text ready for vectorization
    """
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))  - {'not', 'but', 'however', 'no', 'yet'}
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Lemmatization - convert words to their base form
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

def plot_top_words(dataframe, text_column='clean_text', target_column='target', target_value=None, n=25):
    """
    Plot the top N most frequent words as a horizontal bar chart.
    Can filter by target variable (e.g., disaster vs non-disaster)
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing text and target columns
    text_column : str, default='clean_text'
        Name of the column containing cleaned text
    target_column : str, default='target'
        Name of the target column
    target_value : int or str, optional
        Filter by specific target value (e.g., 1 for disaster, 0 for non-disaster)
        If None, uses all data
    n : int, default=25
        Number of top words to display
        
    Returns:
    --------
    None (displays plot)
    """
    # Filter by target if specified
    if target_value is not None:
        data = dataframe[dataframe[target_column] == target_value]
        title_suffix = f' ({target_column}={target_value})'
    else:
        data = dataframe
        title_suffix = ' (All Data)'
    
    # Combine all words from the text series
    all_words = ' '.join(data[text_column]).split()
    
    # Count word frequencies
    word_freq = Counter(all_words)
    
    # Get top N words
    top_words = word_freq.most_common(n)
    
    # Separate words and frequencies
    words = [word for word, freq in top_words]
    frequencies = [freq for word, freq in top_words]
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, 8))
    plt.barh(words, frequencies, color='steelblue')
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.title(f'Top {n} Most Frequent Words{title_suffix}', fontsize=14)
    plt.gca().invert_yaxis()  # Invert y-axis to have highest frequency at top
    plt.tight_layout()
    plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_word_cloud(text):
    """
    Plot a word cloud from the text.
    
    Parameters:
    -----------
    text: list of strings
    
    Returns:
    --------
    None (displays plot)
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def include_numerical_features_and_clean_text(X):
    """
    Include numerical features and clean text in the dataframe.
    Features included:
    - Length of the tweet
    - Word count
    - Number of stop words
    - Number of punctuation characters    
    - Clean text
    
    Parameters:
    -----------
    X: pandas DataFrame
    
    Returns:
    --------
    X: pandas DataFrame
    """
    # Length of the tweet
    X['length'] = X['text'].apply(len)
    
    # Word count
    X['word_count'] = X['text'].apply(lambda x: len(str(x).split()))
    
    # Number of stop words
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    X['num_stop_words'] = X['text'].apply(lambda x: len([word for word in x.split() if word in stop_words]))
    
    # Number of punctuation characters
    X['num_punctuation'] = X['text'].apply(lambda x: len([char for char in x if char in string.punctuation]))
    
    # Clean text
    X['clean_text'] = X['text'].apply(clean_text)
    
    return X

def vectorize_text(X, vectorizer=None):
    """
    Vectorize the text using TF-IDF and combine with numerical features.
    
    This function:
    1. Applies TF-IDF vectorization with unigrams and bigrams (ngram_range=(1,2))
    2. Limits features to top 10,000 most important terms
    3. Extracts numerical features (length, word_count, num_stop_words, num_punctuation)
    4. Horizontally stacks (concatenates) text features with numerical features
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Dataframe with 'clean_text' column and numerical feature columns
    vectorizer : TfidfVectorizer, optional
        Pre-fitted vectorizer object. If None, a new vectorizer will be created and fitted.
    
    Returns:
    --------
    X_transformed : numpy.ndarray
        Combined features (TF-IDF + numerical)
    vectorizer : TfidfVectorizer
        Fitted vectorizer object (useful for transforming other data later)
    """
    # Initialize TF-IDF vectorizer with unigrams and bigrams if not provided
    if vectorizer is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        X_text = vectorizer.fit_transform(X['clean_text']).toarray()
    else:
        X_text = vectorizer.transform(X['clean_text']).toarray()

    # Define numerical feature columns
    numerical_features = ['length', 'word_count', 'num_stop_words', 'num_punctuation']

    # Extract numerical features as numpy arrays
    X_numerical = X[numerical_features].values

    # Combine text features with numerical features horizontally
    X_transformed = np.hstack([X_text, X_numerical])

    return X_transformed


def prepare_single_sentence(sentence, vectorizer):
    """
    Prepare a single sentence for model prediction.
    
    This function takes a raw text sentence and applies the complete preprocessing pipeline:
    1. Converts sentence to DataFrame format
    2. Extracts numerical features (length, word count, stopwords, punctuation)
    3. Cleans the text (removes URLs, mentions, stopwords, lemmatization)
    4. Vectorizes using the provided fitted TF-IDF vectorizer
    5. Combines all features into a prediction-ready numpy array
    
    Parameters:
    -----------
    sentence : str
        Raw text sentence to prepare for prediction
        Example: "There's a huge fire in the building!"
    vectorizer : TfidfVectorizer
        Pre-fitted TF-IDF vectorizer object from training
        Must be the same vectorizer used during model training
    
    Returns:
    --------
    X_transformed : numpy.ndarray
        Processed features ready for model.predict()
        Shape: (1, n_features) where n_features = 10,000 TF-IDF + 4 numerical
    
    Example:
    --------
    >>> # After training
    >>> sentence = "Earthquake hits California!"
    >>> X = prepare_single_sentence(sentence, vectorizer)
    >>> prediction = model.predict(X)
    >>> print("Disaster" if prediction[0] == 1 else "Not Disaster")
    """
    # Step 1: Convert sentence to DataFrame format
    # The preprocessing functions expect a DataFrame with a 'text' column
    df = pd.DataFrame({'text': [sentence]})
    
    # Step 2: Extract numerical features and clean text
    # This adds columns: length, word_count, num_stop_words, num_punctuation, clean_text
    df = include_numerical_features_and_clean_text(df)
    
    # Step 3: Vectorize text and combine with numerical features
    # Uses the provided fitted vectorizer to transform the text
    X_transformed = vectorize_text(df, vectorizer=vectorizer)
    
    return X_transformed
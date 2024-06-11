import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

# Load the data
tweets_df = pd.read_csv('tweets.csv')

# Function to clean the tweet text
def clean_tweet(tweet):
    tweet = tweet.lower()  # Convert to lowercase
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#\w+', '', tweet)  # Remove hashtags
    tweet = re.sub(r'[^a-z\s]', '', tweet)  # Remove punctuation and numbers
    tweet = re.sub(r'\s+', ' ', tweet)  # Remove extra spaces
    return tweet

# Custom transformer for text preprocessing
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [clean_tweet(text) for text in X]

# Keywords for classification
keywords = {
    "MMR vaccine causes autism": ["autism"],
    "MMR vaccine causes polio": ["polio"],
    "MMR vaccine causes paralysis": ["paralysis"],
    "MMR vaccine causes other diseases/side effects": ["side effects", "disease", "effects"],
    "MMR vaccine causes death": ["death"],
    "Safety concerns": ["safety", "concerns"]
}

# Function to label the tweets
def label_tweet(tweet):
    for reason, words in keywords.items():
        if any(word in tweet for word in words):
            return "negative", reason
    if any(word in tweet for word in ["safe", "effective", "prevent", "critical", "save", "recommended", "eradicated"]):
        return "positive", None
    return "unclear", None

# Apply cleaning and labeling
tweets_df['Cleaned_Tweet'] = tweets_df['Tweet'].apply(clean_tweet)
tweets_df[['Stance', 'Reason']] = tweets_df['Cleaned_Tweet'].apply(lambda tweet: pd.Series(label_tweet(tweet)))

# Prepare the data for training
X = tweets_df['Cleaned_Tweet']
y = tweets_df['Stance']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline for preprocessing and model training
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 2))),
    ('classifier', SVC())  # Change the classifier as needed
])

# Define hyperparameters for GridSearchCV
param_grid = {
    'vectorizer__max_features': [1000, 2000, 3000],
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf']
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_model.predict(X_test)
print("Tuned Model Performance:")
print(classification_report(y_test, y_pred_best))

# Function to classify tweets and identify reasons
def classify_tweets(tweets, model):
    cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]
    predictions = model.predict(cleaned_tweets)
    results = []
    for tweet, stance in zip(tweets, predictions):
        reason = None
        if stance == 'negative':
            for reason_category, words in keywords.items():
                if any(word in clean_tweet(tweet) for word in words):
                    reason = reason_category
                    break
        results.append(f"Tweet: {tweet}\nStance: {stance}, Reason: {reason}\n")
    return results

# Classify the tweets
tweets_to_classify = tweets_df['Tweet'].tolist()
classification_results = classify_tweets(tweets_to_classify, best_model)

# Print the results
for result in classification_results:
    print(result)

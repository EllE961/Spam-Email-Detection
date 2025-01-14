import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class SMSClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.vectorizer = None

    def load_data(self):
        df_raw = pd.read_csv(self.data_path, encoding='latin-1')
        df_raw.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
        self.df = df_raw[['label', 'message']].dropna()

    def clean_text(self, txt):
        txt = txt.lower()
        txt = re.sub(r'[^a-z0-9\s]', '', txt)
        tokens = txt.split()
        stop = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stop]
        return ' '.join(tokens)

    def preprocess(self):
        nltk.download('stopwords')
        self.df['message'] = self.df['message'].apply(self.clean_text)

    def split_data(self):
        X = self.df['message']
        y = self.df['label']
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def vectorize_data(self, X_train, X_test):
        self.vectorizer = TfidfVectorizer()
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        return X_train_vec, X_test_vec

    def train_model(self, X_train_vec, y_train):
        self.model = MultinomialNB()
        self.model.fit(X_train_vec, y_train)

    def evaluate_model(self, X_test_vec, y_test):
        preds = self.model.predict(X_test_vec)
        print(classification_report(y_test, preds))
        cf = confusion_matrix(y_test, preds)
        print(cf)
        return preds, cf

    def run_all(self):
        self.load_data()
        self.preprocess()
        X_train, X_test, y_train, y_test = self.split_data()
        X_train_vec, X_test_vec = self.vectorize_data(X_train, X_test)
        self.train_model(X_train_vec, y_train)
        preds, cf = self.evaluate_model(X_test_vec, y_test)
        return self.df, preds, cf

def create_wordclouds(df):
    ham_text = ' '.join(df[df['label'] == 'ham']['message'])
    spam_text = ' '.join(df[df['label'] == 'spam']['message'])
    wc_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
    plt.figure(figsize=(8,4))
    plt.imshow(wc_ham, interpolation='bilinear')
    plt.axis('off')
    plt.title("Ham Word Cloud")
    plt.show()
    wc_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
    plt.figure(figsize=(8,4))
    plt.imshow(wc_spam, interpolation='bilinear')
    plt.axis('off')
    plt.title("Spam Word Cloud")
    plt.show()

def plot_message_length_distribution(df):
    df['msg_len'] = df['message'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8,4))
    sns.histplot(data=df, x='msg_len', hue='label', kde=True, bins=30)
    plt.title("Message Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.show()

def plot_label_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='label')
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

def plot_confusion_matrix(cf, labels=('ham','spam')):
    plt.figure(figsize=(6,4))
    sns.heatmap(cf, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def plot_top_features(model, vectorizer, n=20):
    features = vectorizer.get_feature_names_out()
    spam_coefs = model.feature_log_prob_[1]
    top_idx = np.argsort(spam_coefs)[::-1][:n]
    top_words = features[top_idx]
    top_vals = spam_coefs[top_idx]
    plt.figure(figsize=(8,4))
    sns.barplot(x=top_vals, y=top_words, orient='h')
    plt.title(f"Top {n} Features for Spam")
    plt.xlabel("Log Probability")
    plt.ylabel("Features")
    plt.show()

if __name__ == '__main__':
    classifier = SMSClassifier('spam.csv')
    df_final, preds, cf = classifier.run_all()
    create_wordclouds(df_final)
    plot_message_length_distribution(df_final)
    plot_label_distribution(df_final)
    plot_confusion_matrix(cf)
    plot_top_features(classifier.model, classifier.vectorizer)

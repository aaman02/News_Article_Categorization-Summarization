"""
Utility functions for News Article Classification and Summarization
"""

import pandas as pd
import numpy as np
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class NewsClassifier:
    """News Article Classifier with multiple model support"""
    
    def __init__(self, model_path=None, tfidf_path=None, encoder_path=None, w2v_path=None):
        self.model = None
        self.tfidf = None
        self.encoder = None
        self.w2v_model = None
        self.model_path = model_path
        self.tfidf_path = tfidf_path
        self.encoder_path = encoder_path
        self.w2v_path = w2v_path
        
    def load_models(self):
        """Load all required models and vectorizers"""
        # Load label encoder
        if self.encoder_path:
            self.encoder = joblib.load(self.encoder_path)
        else:
            # Create default encoder for demo
            self.encoder = LabelEncoder()
            self.encoder.classes_ = np.array(['business', 'education', 'entertainment', 'sports', 'technology'])
        
        # Load TF-IDF vectorizer
        if self.tfidf_path:
            self.tfidf = joblib.load(self.tfidf_path)
        
        # Load Word2Vec model
        if self.w2v_path:
            self.w2v_model = Word2Vec.load(self.w2v_path)
        
        # Load classification model
        if self.model_path:
            self.model = joblib.load(self.model_path)
        
        return self
    
    def preprocess_text(self, text):
        """Preprocess text for prediction"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        words = [w for w in words if w not in stop_words]
        text = ' '.join(words)
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        protected_words = {"rs", "mr", "mrs", "dr"}
        tokens = [
            word if word.lower() in protected_words else lemmatizer.lemmatize(word)
            for word in text.split()
            if word.isalpha() and len(word) > 1
        ]
        
        return ' '.join(tokens)
    
    def predict(self, text, use_tfidf=True):
        """Predict category for a single text"""
        if self.model is None:
            return "Model not loaded"
        
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        if use_tfidf and self.tfidf:
            # Use TF-IDF features
            X = self.tfidf.transform([processed_text])
        else:
            # Use Word2Vec features
            words = processed_text.split()
            embeddings = []
            for word in words:
                if word in self.w2v_model.wv:
                    embeddings.append(self.w2v_model.wv[word])
            
            if len(embeddings) > 0:
                X = np.mean(embeddings, axis=0).reshape(1, -1)
            else:
                X = np.zeros((1, 100))  # Default 100-dim embedding
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # If model outputs encoded label, decode it
        if isinstance(prediction, (int, np.integer)):
            prediction = self.encoder.inverse_transform([prediction])[0]
        
        return prediction
    
    def predict_proba(self, text, use_tfidf=True):
        """Get prediction probabilities"""
        if self.model is None:
            return None
        
        processed_text = self.preprocess_text(text)
        
        if use_tfidf and self.tfidf:
            X = self.tfidf.transform([processed_text])
        else:
            words = processed_text.split()
            embeddings = []
            for word in words:
                if word in self.w2v_model.wv:
                    embeddings.append(self.w2v_model.wv[word])
            
            if len(embeddings) > 0:
                X = np.mean(embeddings, axis=0).reshape(1, -1)
            else:
                X = np.zeros((1, 100))
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[0]
        else:
            return None


class NewsSummarizer:
    """News Article Summarizer using extractive and abstractive methods"""
    
    def __init__(self):
        self.max_summary_length = 60  # words
    
    def extractive_summary(self, text, num_sentences=3):
        """Generate extractive summary by selecting important sentences"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return "No content available for summarization."
        
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Word frequency scoring
        word_freq = {}
        words = nltk.word_tokenize(text.lower())
        words = [w for w in words if w.isalnum() and w not in stopwords.words('english')]
        
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            score = 0
            sent_words = nltk.word_tokenize(sentence.lower())
            for word in sent_words:
                if word in word_freq:
                    score += word_freq[word]
            sentence_scores[i] = score / (len(sent_words) + 1)  # Normalize by length
        
        # Get top sentences
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        top_indices = sorted(top_indices)  # Keep original order
        
        summary = ' '.join([sentences[i] for i in top_indices])
        
        # Truncate to max length
        words = summary.split()
        if len(words) > self.max_summary_length:
            summary = ' '.join(words[:self.max_summary_length]) + '...'
        
        return summary
    
    def headline_summary(self, text):
        """Generate a headline-style summary (very concise)"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return "No headline available"
        
        # Get first sentence as base
        sentences = nltk.sent_tokenize(text)
        if len(sentences) == 0:
            return "No headline available"
        
        first_sentence = sentences[0]
        words = first_sentence.split()
        
        # Truncate to ~15 words for headline
        if len(words) > 15:
            headline = ' '.join(words[:15]) + '...'
        else:
            headline = first_sentence
        
        return headline
    
    def generate_summary(self, text, style='concise'):
        """
        Generate summary based on style
        
        Args:
            text: Input text to summarize
            style: 'concise' (60 words) or 'headline' (title style)
        
        Returns:
            Summary text
        """
        if style == 'headline':
            return self.headline_summary(text)
        else:
            return self.extractive_summary(text)


def load_sample_data():
    """Load sample news data for demo"""
    sample_news = {
        'Sports': """The championship game ended in a thrilling victory as the home team scored 
        a last-minute goal. The crowd erupted in celebration as players rushed the field. 
        This marks the team's first championship win in over a decade, bringing joy to 
        millions of fans worldwide. The star player was named MVP after scoring two goals.""",
        
        'Technology': """A groundbreaking artificial intelligence system has been unveiled by 
        researchers, capable of solving complex mathematical problems faster than ever before. 
        The new AI model uses advanced neural networks and has shown remarkable accuracy in 
        various benchmarks. Tech companies are already exploring applications in healthcare 
        and finance.""",
        
        'Business': """Stock markets reached record highs today as investor confidence surged 
        following positive economic data. Major tech companies led the rally with significant 
        gains. Analysts predict continued growth in the coming quarters as consumer spending 
        remains strong and unemployment rates stay low.""",
        
        'Education': """A new study reveals that online learning has become increasingly 
        effective, with students showing improved engagement and retention rates. Universities 
        are adopting hybrid models that combine traditional classroom teaching with digital 
        tools. Educators emphasize the importance of personalized learning experiences.""",
        
        'Entertainment': """The highly anticipated movie premiere drew celebrities and fans 
        alike to the red carpet event. Critics are praising the film's innovative storytelling 
        and stunning visuals. The director's latest work is expected to be a major contender 
        in the upcoming awards season."""
    }
    
    return sample_news


# Initialize default instances
default_classifier = NewsClassifier()
default_summarizer = NewsSummarizer()

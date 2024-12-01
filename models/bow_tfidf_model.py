from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

class BoWTfidfClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = make_pipeline(self.vectorizer, MLPClassifier(hidden_layer_sizes=(100,), max_iter=500))
    
    def train(self, reviews, labels):
        self.classifier.fit(reviews, labels)
    
    def predict(self, reviews):
        return self.classifier.predict(reviews)

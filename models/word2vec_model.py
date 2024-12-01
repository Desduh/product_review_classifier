from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class Word2VecClassifier:
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.word2vec_model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )
        
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500))
        ])
    
    def train(self, reviews, labels):
        if isinstance(reviews[0], str): 
            tokenized_reviews = [review.split() for review in reviews]
        else:
            tokenized_reviews = reviews

        self.word2vec_model = Word2Vec(tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)
        
        X = self.vectorize_reviews(tokenized_reviews, self.word2vec_model)
        
        self.classifier.fit(X, labels)

    def get_review_vector(self, review):
        return [self.word2vec_model.wv[word] for word in review if word in self.word2vec_model.wv]
    
    def predict(self, reviews):
        if isinstance(reviews[0], str):  
            tokenized_reviews = [review.split() for review in reviews]
        else:
            tokenized_reviews = reviews

        X = self.vectorize_reviews(tokenized_reviews, self.word2vec_model)
        return self.classifier.predict(X)

    def vectorize_reviews(self, reviews, word2vec_model):
        vectorized_reviews = []
        for review in reviews:
            word_vectors = [
                word2vec_model.wv[word] for word in review if word in word2vec_model.wv
            ]
            if word_vectors:
                vectorized_reviews.append(np.mean(word_vectors, axis=0))
            else:
                vectorized_reviews.append(np.zeros(word2vec_model.vector_size))
        return np.array(vectorized_reviews)

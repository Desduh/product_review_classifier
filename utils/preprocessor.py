import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Remove caracteres especiais e converte para minúsculas
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenização
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

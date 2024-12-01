import nltk
nltk.download('stopwords')  
nltk.download('punkt')      
import pandas as pd
from utils.preprocessor import preprocess_text
from models.word2vec_model import Word2VecClassifier
from models.bow_tfidf_model import BoWTfidfClassifier
from utils.metrics import generate_confusion_matrix, calculate_metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados
train_data = pd.read_csv('data/reviews_train.csv')
test_data = pd.read_csv('data/reviews_test.csv')

# Pré-processamento
train_reviews = train_data['review'].apply(preprocess_text)
test_reviews = test_data['review'].apply(preprocess_text)
train_labels = train_data['label']
test_labels = test_data['label']

# Modelo Word2Vec
w2v_model = Word2VecClassifier()
w2v_model.train(train_reviews, train_labels)
w2v_preds = w2v_model.predict(test_reviews)

# Modelo BOW-TFIDF
bow_model = BoWTfidfClassifier()
bow_model.train(train_data['review'], train_labels)
bow_preds = bow_model.predict(test_data['review'])

# Matrizes de confusão
labels = ['positive', 'negative', 'neutral']
w2v_conf_matrix = generate_confusion_matrix(test_labels, w2v_preds, labels)
bow_conf_matrix = generate_confusion_matrix(test_labels, bow_preds, labels)

# Calcular as métricas
metrics = calculate_metrics(w2v_conf_matrix, labels)
print("\nMetric calculate - Word2Vec:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

metrics = calculate_metrics(bow_conf_matrix, labels)
print("\nMetric calculate - BOW-TFIDF:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")


# Função para exibir e salvar a matriz de confusão
def plot_confusion_matrix(matrix, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.show()

# Gerar as imagens da matriz de confusão
plot_confusion_matrix(w2v_conf_matrix, 'Confusion Matrix - Word2Vec', 'confusion_matrix_w2v.png')
plot_confusion_matrix(bow_conf_matrix, 'Confusion Matrix - BOW-TFIDF', 'confusion_matrix_bow.png')
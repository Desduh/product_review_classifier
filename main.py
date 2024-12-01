import nltk
nltk.download('stopwords')  
nltk.download('punkt')      
import pandas as pd
from utils.preprocessor import preprocess_text
from models.word2vec_model import Word2VecClassifier
from models.bow_tfidf_model import BoWTfidfClassifier
from utils.metrics import generate_confusion_matrix, calculate_metrics
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np

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

# Função para exibir e salvar a matriz de confusão com colormap de verde para vermelho
def plot_confusion_matrix(matrix, title, filename, labels):
    # Criando um colormap customizado para a diagonal (vermelho-verde) e os não-diagonais (verde-vermelho)
    cmap_diag = LinearSegmentedColormap.from_list("diag_red_green", ["#3EA201", "#295700"])  # Para a diagonal
    cmap_offdiag = LinearSegmentedColormap.from_list("offdiag_green_red", ["white", "red"])  # Para os não-diagonais

    plt.figure(figsize=(8, 6))

    # Criando uma matriz modificada como tipo float para permitir o uso de NaN
    modified_matrix = matrix.astype(np.float32)
    
    # Criando uma cópia da matriz para aplicar NaN nos valores fora da diagonal
    off_diag_matrix = np.copy(modified_matrix)
    np.fill_diagonal(off_diag_matrix, np.nan)  # Definindo NaN na diagonal

    # Plotando a matriz fora da diagonal com o colormap verde-vermelho
    sns.heatmap(off_diag_matrix, annot=True, cmap=cmap_offdiag, mask=np.isnan(off_diag_matrix), 
                xticklabels=labels, yticklabels=labels, cbar=False, square=True, vmin=0)

    # Criando uma máscara para a diagonal e fora da diagonal
    diag_mask = np.eye(matrix.shape[0], dtype=bool)

    # Plotando a diagonal com o colormap vermelho-verde
    sns.heatmap(modified_matrix, annot=True, fmt='.2f', cmap=cmap_diag, mask=~diag_mask, 
                xticklabels=labels, yticklabels=labels, cbar=False, square=True)

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.show()


# Gerar as imagens da matriz de confusão com colormap verde-vermelho
plot_confusion_matrix(w2v_conf_matrix, 'Confusion Matrix - Word2Vec', 'confusion_matrix_w2v.png', labels)
plot_confusion_matrix(bow_conf_matrix, 'Confusion Matrix - BOW-TFIDF', 'confusion_matrix_bow.png', labels)
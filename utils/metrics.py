import numpy as np

def generate_confusion_matrix(true_labels, predicted_labels, labels):
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(true_labels, predicted_labels):
        true_idx = labels.index(true)
        pred_idx = labels.index(pred)
        confusion_matrix[true_idx, pred_idx] += 1
    return confusion_matrix

def calculate_metrics(conf_matrix, labels):
    # Extrair os valores da matriz de confusão
    print(conf_matrix)
    TP = conf_matrix[0][0]  # True Positive
    FNP = conf_matrix[1][0]  # False Negative Positive
    FNeP = conf_matrix[2][0]  # False Positive Neutral
    FPN = conf_matrix[0][1]  # False Positive Negative
    TN = conf_matrix[1][1]  # True Negative
    FNeN = conf_matrix[2][1]  # False Neutral Negative
    FPNe = conf_matrix[0][2]  # False Positive Neutral
    FNNe = conf_matrix[1][2]  # False Negative Neutral
    TNe = conf_matrix[2][2]  # True Neutral

    T = TP + FNP + FNeP + FPN + TN + FNeN + FPNe + FNNe + TNe  # Total
    
    # Calculando as métricas
    P = TP + FPN + FPNe
    N = TN + FNP + FNNe
    Ne = TNe + FNeN + FNeP

    TPR = TP / P  # True Positive Rate
    FPNR = FPN / P  # False Positive Negative Rate
    FNeP = FPNe / P  # False Positive Neutral Rate
    FPR = (FNP + FNeP) / (N + Ne)  # False Positive Rate
    TNR = TN / N  # True Negative Rate
    FNP = FNP / N  # False Negative Positive Rate
    FNNe = FNNe / N  # False Negative Neutral Rate
    FNR = (FPN + FNeN) / (P + Ne)  # False Negative Rate
    TNeR = TNe / Ne  # True Neutral Rate
    FNeP = FNeP / Ne  # False Neutral Positive Rate
    FNeN = FNeN / Ne  # False Neutral Negative Rate
    FNeR = (FPNe + FNNe) / (P + N)  # False Neutral Rate

    accuracy = (TP + TN + TNe) / T  # Accuracy
    likelihood_ratio_positive = TPR / FPR  # Likelihood ratio for positive samples
    likelihood_ratio_negative = TNR / FNR  # Likelihood ratio for negative samples
    likelihood_ratio_neutral = TNeR / FNeR  # Likelihood ratio for neutral samples
    
    return {
        "T": f"{T:.2f} (Total samples)",
        "TPR": f"{TPR:.2f} (True Positive Rate)",
        "FPNR": f"{FPNR:.2f} (False Positive Negative Rate)",
        "FNeP": f"{FNeP:.2f} (False Positive Neutral Rate)",
        "FPR": f"{FPR:.2f} (False Positive Rate)",
        "TNR": f"{TNR:.2f} (True Negative Rate)",
        "FNP": f"{FNP:.2f} (False Negative Positive Rate)",
        "FNNe": f"{FNNe:.2f} (False Negative Neutral Rate)",
        "FNR": f"{FNR:.2f} (False Negative Rate)",
        "TNeR": f"{TNeR:.2f} (True Neutral Rate)",
        "FNeP": f"{FNeP:.2f} (False Neutral Positive Rate)",
        "FNeN": f"{FNeN:.2f} (False Neutral Negative Rate)",
        "FNeR": f"{FNeR:.2f} (False Neutral Rate)",
        "accuracy": f"{accuracy:.2f} (Accuracy)",
        "likelihood_ratio_positive": f"{likelihood_ratio_positive:.2f} (Likelihood Ratio for Positive Samples)",
        "likelihood_ratio_negative": f"{likelihood_ratio_negative:.2f} (Likelihood Ratio for Negative Samples)",
        "likelihood_ratio_neutral": f"{likelihood_ratio_neutral:.2f} (Likelihood Ratio for Neutral Samples)"
    }

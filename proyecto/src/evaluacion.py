import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)


def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa un modelo de clasificación y devuelve métricas principales.
    """
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, output_dict=False)
    matriz = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "classification_report": reporte,
        "confusion_matrix": matriz,
        "y_pred": y_pred
    }


def comparar_modelos(resultados):
    """
    Recibe una lista de diccionarios con nombre del modelo y accuracy.
    Devuelve un DataFrame comparativo.
    """
    comparacion = pd.DataFrame(resultados)
    comparacion = comparacion.sort_values(by="Accuracy", ascending=False)
    return comparacion


def evaluar_clustering(X_scaled, labels):
    """
    Evalúa el clustering usando silhouette score.
    """
    score = silhouette_score(X_scaled, labels)
    return score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


def entrenar_arbol_decision(X_train, y_train, max_depth=5, random_state=42):
    """
    Entrena un modelo de Árbol de Decisión para clasificación.
    """
    modelo = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )
    modelo.fit(X_train, y_train)
    return modelo


def entrenar_regresion_logistica(X_train, y_train, random_state=42):
    """
    Entrena un modelo de Regresión Logística para clasificación multiclase.
    """
    modelo = LogisticRegression(
        max_iter=1000,
        random_state=random_state
    )
    modelo.fit(X_train, y_train)
    return modelo


def entrenar_kmeans(X_train_scaled, n_clusters=3, random_state=42):
    """
    Entrena un modelo K-Means para clustering.
    """
    modelo = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    modelo.fit(X_train_scaled)
    return modelo
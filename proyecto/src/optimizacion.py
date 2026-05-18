from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def optimizar_arbol_grid(X_train, y_train, cv=5):
    """
    Optimiza un Árbol de Decisión usando GridSearchCV.
    """
    modelo = DecisionTreeClassifier(random_state=42)

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def optimizar_arbol_random(X_train, y_train, cv=5, n_iter=10):
    """
    Optimiza un Árbol de Decisión usando RandomizedSearchCV.
    """
    modelo = DecisionTreeClassifier(random_state=42)

    param_dist = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6]
    }

    random_search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    return random_search
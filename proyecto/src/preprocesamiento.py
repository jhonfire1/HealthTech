import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cargar_datos(ruta):
    """
    Carga el dataset desde un archivo CSV.
    """
    df = pd.read_csv(ruta)
    return df


def explorar_datos(df):
    """
    Muestra información general del dataset.
    """
    print("Primeras filas:")
    print(df.head())

    print("\nDimensiones del dataset:")
    print(df.shape)

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores nulos:")
    print(df.isnull().sum())

    print("\nDistribución de la variable objetivo:")
    print(df["target_diagnostico"].value_counts())


def preprocesar_datos(df):
    """
    Preprocesa el dataset:
    - Codifica la variable categórica consumo_tabaco
    - Separa variables predictoras y variable objetivo
    """
    df = df.copy()

    df["consumo_tabaco"] = df["consumo_tabaco"].map({
        "Nunca": 0,
        "Ex-fumador": 1,
        "Fumador": 2
    })

    X = df.drop("target_diagnostico", axis=1)
    y = df["target_diagnostico"]

    return X, y


def dividir_datos(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos en entrenamiento y prueba.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def escalar_datos(X_train, X_test):
    """
    Escala los datos numéricos usando StandardScaler.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
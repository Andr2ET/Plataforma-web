import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
import joblib

# Función para extraer meta-features de un dataset
def extraer_meta_features(df):
    meta = {}
    meta["num_filas"] = df.shape[0]
    meta["num_columnas"] = df.shape[1]
    meta["porc_nan"] = df.isna().mean().mean()

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    meta["num_variables_numericas"] = len(num_cols)
    meta["num_variables_categoricas"] = len(cat_cols)

    if len(num_cols) > 0:
        meta["media_num"] = df[num_cols].mean().mean()
        meta["std_num"] = df[num_cols].std().mean()
        meta["skew_num"] = df[num_cols].apply(skew, nan_policy='omit').mean()
        meta["kurtosis_num"] = df[num_cols].apply(kurtosis, nan_policy='omit').mean()
    else:
        meta["media_num"] = 0
        meta["std_num"] = 0
        meta["skew_num"] = 0
        meta["kurtosis_num"] = 0

    if len(cat_cols) > 0:
        meta["cardinalidad_cat"] = df[cat_cols].nunique().mean()
    else:
        meta["cardinalidad_cat"] = 0

    return meta

# Función para simular datasets (ejemplo)
def simular_datasets():
    # Simula datasets con distintas características y etiquetas
    datasets = []
    etiquetas = []

    for i in range(100):
        n_filas = np.random.randint(50, 5000)
        n_cols = np.random.randint(3, 30)
        # Dataset simulado: numérico puro con ruido
        df = pd.DataFrame(np.random.randn(n_filas, n_cols), columns=[f"c{i}" for i in range(n_cols)])
        # Etiqueta por regla simple: clasificación o regresión + pocas/muchas filas
        if n_filas < 1000:
            etiqueta = "LogisticRegression" if i % 2 == 0 else "LinearRegression"
        else:
            etiqueta = "RandomForestClassifier" if i % 2 == 0 else "RandomForestRegressor"
        datasets.append(df)
        etiquetas.append(etiqueta)

    return datasets, etiquetas

def preparar_datos_meta(datasets, etiquetas, tipos_problema):
    datos_meta = []
    for df, etiqueta, tipo in zip(datasets, etiquetas, tipos_problema):
        meta = extraer_meta_features(df)
        meta["tipo_problema"] = tipo
        meta["mejor_modelo"] = etiqueta
        datos_meta.append(meta)
    return pd.DataFrame(datos_meta)

def main():
    # Simular datasets y etiquetas
    datasets, etiquetas = simular_datasets()
    # Asumir tipo_problema según etiqueta para demo
    tipos = ["clasificación" if "Classifier" in e else "regresión" for e in etiquetas]

    df_meta = preparar_datos_meta(datasets, etiquetas, tipos)

    # One-hot encode de 'tipo_problema'
    df_meta_enc = pd.get_dummies(df_meta, columns=["tipo_problema"])

    X = df_meta_enc.drop(columns=["mejor_modelo"])
    y = df_meta_enc["mejor_modelo"]

    modelo_meta = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_meta.fit(X, y)

    joblib.dump(modelo_meta, "meta_model.pkl")
    print("Modelo entrenado y guardado como meta_model.pkl")

if __name__ == "__main__":
    main()

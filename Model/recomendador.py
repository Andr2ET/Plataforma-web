import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import joblib

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

class RecomendadorAlgoritmo:
    def __init__(self, ruta_modelo="meta_model.pkl"):
        self.modelo = joblib.load(ruta_modelo)

    def recomendar(self, df, tipo_problema):
        meta = extraer_meta_features(df)
        meta["tipo_problema"] = tipo_problema

        X_input = pd.DataFrame([meta])
        X_input = pd.get_dummies(X_input, columns=["tipo_problema"])

        # Asegurar columnas que el modelo espera
        for col in self.modelo.feature_names_in_:
            if col not in X_input.columns:
                X_input[col] = 0

        X_input = X_input[self.modelo.feature_names_in_]

        prediccion = self.modelo.predict(X_input)[0]
        return prediccion

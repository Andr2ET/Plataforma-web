# ml_engine.py
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def obtener_modelos_disponibles(tipo):
    """
    Devuelve un diccionario de modelos disponibles según el tipo de problema.
    """
    if tipo == "clasificacion":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier()
        }
    elif tipo == "regresion":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor()
        }
    else:
        raise ValueError(f"Tipo de modelo no soportado: {tipo}")

def obtener_metricas(tipo):
    """
    Devuelve un diccionario de métricas disponibles según el tipo de problema.
    """
    if tipo == "clasificacion":
        return {
            "Accuracy": accuracy_score,
            "Precision": precision_score,
            "Recall": recall_score,
            "F1-Score": f1_score
        }
    elif tipo == "regresion":
        return {
            "MSE": mean_squared_error,
            "MAE": mean_absolute_error,
            "R2": r2_score
        }
    else:
        raise ValueError(f"Tipo de métrica no soportado: {tipo}")

def entrenar_modelos(seleccionados, modelos_dict, X_train, X_test, y_train, y_test, tipo):
    """
    Entrena y evalúa los modelos seleccionados. Devuelve resultados con métricas y predicciones.
    """
    resultados = {}
    metricas = obtener_metricas(tipo)

    for nombre in seleccionados:
        modelo = modelos_dict[nombre]
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        info = {}
        for met_nombre, met_func in metricas.items():
            if tipo == "clasificacion" and met_nombre in ["Precision", "Recall", "F1-Score"]:
                valor = met_func(y_test, pred, average="weighted")
            else:
                valor = met_func(y_test, pred)
            info[met_nombre] = round(valor, 4)

        resultados[nombre] = {
            "metricas": info,
            "resultados": {
                "real": y_test.reset_index(drop=True).tolist(),
                "predicho": pred.tolist()
            },
            "modelo_entrenado": modelo
        }

    return resultados

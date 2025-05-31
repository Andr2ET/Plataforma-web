# preprocessor.py
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Realiza limpieza del dataset:
    - Elimina columnas vacías
    - Elimina filas con NaN
    - Elimina columnas con un solo valor
    - Elimina valores negativos
    Retorna el DataFrame limpio y un reporte detallado.
    """
    report = {}

    # 1. Eliminar columnas completamente vacías
    empty_cols = df.columns[df.isnull().all()].tolist()
    df.drop(columns=empty_cols, inplace=True)
    report["Columnas vacías eliminadas"] = empty_cols

    # 2. Eliminar filas con valores nulos
    rows_before = df.shape[0]
    df.dropna(inplace=True)
    rows_after = df.shape[0]
    report["Filas con NaN eliminadas"] = rows_before - rows_after

    # 3. Eliminar columnas con un solo valor (irrelevantes)
    single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=single_value_cols, inplace=True)
    report["Columnas irrelevantes eliminadas"] = single_value_cols

    # 4. Eliminar valores negativos en columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            df = df[df[col] >= 0]
            report[f"Valores negativos eliminados en '{col}'"] = int(negative_count)

    return df.reset_index(drop=True), report

import pandas as pd
from io import BytesIO

def load_dataset(uploaded_file):
    if uploaded_file is None:
        return None, "No file uploaded."

    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Formato de archivo no soportado. Usa CSV o Excel."
        return df, None
    except Exception as e:
        return None, f"Error al leer el archivo: {e}"

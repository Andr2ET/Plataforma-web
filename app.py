import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.ml_engine import obtener_modelos_disponibles, entrenar_modelos
from utils.preprocessor import clean_dataset
import joblib
import io
import base64
from Model.recomendador import RecomendadorAlgoritmo

st.set_page_config(layout="wide")
st.title("🧠 Sistema Inteligente de Entrenamiento de Modelos")

# Paso 1: Cargar datos
st.header("📁 Paso 1: Cargar archivo CSV o XSLX")
archivo = st.file_uploader("Sube tu archivo CSV o XSLX", type=["csv", "xlsx"])

if archivo:
    if archivo.name.endswith(".csv"):
        try:
            df = pd.read_csv(archivo, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(archivo, encoding='latin1')
    else:
        df = pd.read_excel(archivo)
    
    df, reporte = clean_dataset(df)
    st.write("🧽 Limpieza aplicada:", reporte)
    st.dataframe(df.head())
    st.session_state["data"] = df


if "data" in st.session_state:
    st.header("🔧 Paso 2: Seleccionar variables")
    df = st.session_state["data"]
    columnas = df.columns.tolist()

    col1, col2 = st.columns(2)  # Crear 2 columnas iguales

    with col1:
        inputs = st.multiselect("Selecciona las variables de entrada (X)", columnas)

    with col2:
        target = st.selectbox("Selecciona la variable objetivo (y)", columnas)

    if inputs and target:
        st.session_state["inputs"] = inputs
        st.session_state["target"] = target


# Paso 3: Tipo de problema
if "inputs" in st.session_state and "target" in st.session_state:
    st.header("📂 Paso 3: Selecciona el tipo de problema")
    tipo_problema = st.radio("¿Qué tipo de problema deseas resolver?", ["clasificacion", "regresion"])
    st.session_state["tipo_problema"] = tipo_problema

# Paso 4 Seleccionar modelos y entrenar
if "tipo_problema" in st.session_state:
    st.header("🤖 Paso 4: Entrenar modelos")

    tipo = st.session_state["tipo_problema"]
    modelos_disponibles = obtener_modelos_disponibles(tipo)
    nombres_modelos = list(modelos_disponibles.keys())

    # Recomendación
    recomendador = RecomendadorAlgoritmo()
    modelo_recomendado = recomendador.recomendar(df[st.session_state["inputs"] + [st.session_state["target"]]], tipo)
    st.info(f"✅ Modelo recomendado por el sistema: **{modelo_recomendado}**")

    modelos_seleccionados = st.multiselect("Selecciona los modelos a entrenar", nombres_modelos, default=[modelo_recomendado])

    if modelos_seleccionados:
        X = df[st.session_state["inputs"]]
        y = df[st.session_state["target"]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        resultados = entrenar_modelos(modelos_seleccionados, modelos_disponibles, X_train, X_test, y_train, y_test, tipo)

        st.success("✅ Modelos entrenados correctamente")
        for nombre, info in resultados.items():
            st.subheader(f"📊 Resultados para {nombre}")
            st.write("🔢 Métricas:", info["metricas"])
            st.write("📈 Comparación real vs. predicho")
            st.dataframe(pd.DataFrame(info["resultados"]))

# Paso 5: Ver predicciones
if "modelos_entrenados" in st.session_state:
    st.header("📊 Paso 5: Resultados de predicción")

    modelos_entrenados = st.session_state["modelos_entrenados"]
    modelo_seleccionado = st.selectbox("Selecciona un modelo para visualizar sus resultados", list(modelos_entrenados.keys()))

    if modelo_seleccionado:
        info = modelos_entrenados[modelo_seleccionado]
        resultados_df = pd.DataFrame({
            "real": info["resultados"]["real"],
            "predicho": info["resultados"]["predicho"]
        })
        st.dataframe(resultados_df.head(10))

# Paso 6: Exportar modelos
if "modelos_entrenados" in st.session_state:
    st.header("📥 Paso 6: Exportar y descargar modelos entrenados")

    modelos_entrenados = st.session_state["modelos_entrenados"]
    modelo_para_descargar = st.selectbox("Selecciona un modelo para exportar", list(modelos_entrenados.keys()))

    if modelo_para_descargar:
        modelo_objeto = st.session_state["modelos_entrenados"][modelo_para_descargar].get("modelo_entrenado")
        
        if not modelo_objeto:
            st.warning("⚠️ El objeto del modelo no fue almacenado, actualizando almacenamiento...")
            # Guardar el modelo entrenado si aún no se guardó
            df = st.session_state["data"]
            inputs = st.session_state["inputs"]
            target = st.session_state["target"]
            X = df[inputs]
            y = df[target]
            from sklearn.model_selection import train_test_split
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

            # Volver a entrenar para guardarlo
            modelo_real = obtener_modelos_disponibles(st.session_state["tipo_problema"])[modelo_para_descargar]
            modelo_real.fit(X_train, y_train)
            st.session_state["modelos_entrenados"][modelo_para_descargar]["modelo_entrenado"] = modelo_real
            modelo_objeto = modelo_real

        # Convertir a bytes y permitir descarga
        buffer = io.BytesIO()
        joblib.dump(modelo_objeto, buffer)
        buffer.seek(0)

        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{modelo_para_descargar}.pkl">📦 Descargar modelo `{modelo_para_descargar}`</a>'
        st.markdown(href, unsafe_allow_html=True)
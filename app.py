import streamlit as st
from utils.data_loader import load_dataset

st.set_page_config(page_title="Plataforma ML", layout="wide")

st.title("🔍 Plataforma Predictiva con ML")
st.subheader("Paso 1: Subida y vista previa del dataset")

uploaded_file = st.file_uploader("Sube tu dataset (.csv o .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    df, error = load_dataset(uploaded_file)

    if error:
        st.error(error)
    else:
        st.success("¡Archivo cargado correctamente!")
        st.write(f"Forma del dataset: {df.shape}")
        st.dataframe(df.head(10))  # Vista previa
        st.write("Tipos de columnas detectadas:")
        st.write(df.dtypes)
else:
    st.info("Por favor, sube un archivo para comenzar.")

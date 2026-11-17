"""
Dashboard del examen de medio ciclo: predicción de enfermedad cardíaca con los cuatro algoritmos.
Grupo 4 - Minería de Datos (CRISP-ML(Q), Fase 6: Despliegue)

Ejecución:  python -m streamlit run app_examen.py
Archivos necesarios (los genera el notebook y van en la misma carpeta):
  scaler.joblib, modelo_rf1.joblib, modelo_rf2.joblib, modelo_lr.joblib, pesos_mlp.npz,
  info_modelos.json, comparativa_examen.csv, requirements.txt
  y, opcionalmente, roc_curvas.png, curvas_aprendizaje.png y matrices_confusion.png
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

CARPETA = os.path.dirname(os.path.abspath(__file__))
VARIABLES = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral',
             'fasting_blood_sugar', 'electrocardiographic', 'max_heart_rate',
             'induced_angina', 'ST_depression', 'slope', 'no_of_vessels', 'thal']

st.set_page_config(page_title="Predicción de enfermedad cardíaca", page_icon="❤️", layout="wide")


def ruta(nombre):
    return os.path.join(CARPETA, nombre)


@st.cache_resource
def cargar_modelos():
    with open(ruta("info_modelos.json"), encoding="utf-8") as f:
        info = json.load(f)
    escalador = joblib.load(ruta("scaler.joblib"))
    modelos = {}
    for m in info["modelos"]:
        if m["formato"] == "mlp":
            pesos = np.load(ruta(m["archivo"]))
            modelos[m["nombre"]] = [pesos[f"arr_{i}"] for i in range(len(pesos.files))]   # [W1, b1, W2, b2, ...]
        else:
            modelos[m["nombre"]] = joblib.load(ruta(m["archivo"]))
    return info, escalador, modelos


def probabilidad(formato, modelo, x):
    """Devuelve P(enfermo) para un dato ya estandarizado."""
    if formato == "mlp":
        a = x
        n_capas = len(modelo) // 2
        for i in range(n_capas):
            a = a @ modelo[2 * i] + modelo[2 * i + 1]
            a = 1 / (1 + np.exp(-a)) if i == n_capas - 1 else np.maximum(a, 0)   # sigmoide en la salida, ReLU en las ocultas
        return float(a[0, 0])
    return float(modelo.predict_proba(x)[0, 1])


if not all(os.path.exists(ruta(a)) for a in ["info_modelos.json", "scaler.joblib"]):
    st.error("Faltan archivos del modelo. Sube a la misma carpeta los archivos que genera el notebook.")
    st.stop()

info, escalador, modelos = cargar_modelos()
fichas = {m["nombre"]: m for m in info["modelos"]}
nombres = list(fichas.keys())
mejor = info["mejor"]

st.title("❤️ Predicción de enfermedad cardíaca: machine learning y deep learning")
st.caption("Caso de estudio: Mishra et al. (2024) · Dataset Cleveland (UCI) · CRISP-ML(Q), Fase 6: Despliegue")

with st.sidebar:
    st.header("Algoritmo para predecir")
    elegido = st.selectbox("Algoritmo", options=nombres, index=nombres.index(mejor))
    ficha = fichas[elegido]
    met = ficha["metricas_prueba"]
    st.write(f"**Tipo:** {ficha['tipo_algoritmo']}")
    st.write(f"**Exactitud en prueba:** {met['Exactitud']:.2%}")
    st.write(f"**Recall en prueba:** {met['Recall']:.3f}")
    st.write(f"**F1 en prueba:** {met['F1']:.3f}")
    st.write(f"**AUC en prueba:** {met['AUC']:.3f}")
    st.caption(f"Mejor algoritmo según la regla definida: {mejor}.")
    st.warning("Uso académico. No sustituye un diagnóstico médico.")

tab_pred, tab_comp = st.tabs(["Predicción", "Comparativa de algoritmos"])

# ---- Pestaña 1: predicción de un paciente nuevo ----
with tab_pred:
    st.subheader("Datos clínicos del paciente")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Edad (años)", min_value=20, max_value=90, value=55)
        sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: {0: "Mujer", 1: "Hombre"}[x])
        chest_pain = st.selectbox(
            "Tipo de dolor de pecho", options=[1, 2, 3, 4],
            format_func=lambda x: {1: "1 - Angina típica", 2: "2 - Angina atípica",
                                   3: "3 - Dolor no anginoso", 4: "4 - Asintomático"}[x])
        blood_pressure = st.number_input("Presión arterial en reposo (mm Hg)", min_value=80, max_value=220, value=130)
        serum_cholestoral = st.number_input("Colesterol (mg/dl)", min_value=100, max_value=600, value=245)
    with c2:
        fasting_blood_sugar = st.selectbox(
            "Azúcar en ayunas > 120 mg/dl", options=[0, 1], format_func=lambda x: {0: "No", 1: "Sí"}[x])
        electrocardiographic = st.selectbox(
            "Electrocardiograma en reposo", options=[0, 1, 2],
            format_func=lambda x: {0: "0 - Normal", 1: "1 - Anomalía ST-T", 2: "2 - Hipertrofia"}[x])
        max_heart_rate = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=220, value=150)
        induced_angina = st.selectbox(
            "Angina inducida por ejercicio", options=[0, 1], format_func=lambda x: {0: "No", 1: "Sí"}[x])
    with c3:
        ST_depression = st.number_input("Depresión ST por ejercicio", min_value=0.0, max_value=7.0,
                                        value=1.0, step=0.1)
        slope = st.selectbox(
            "Pendiente del segmento ST", options=[1, 2, 3],
            format_func=lambda x: {1: "1 - Ascendente", 2: "2 - Plana", 3: "3 - Descendente"}[x])
        no_of_vessels = st.selectbox("Vasos principales coloreados (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thal", options=[3, 6, 7],
            format_func=lambda x: {3: "3 - Normal", 6: "6 - Defecto fijo", 7: "7 - Defecto reversible"}[x])

    if st.button("Predecir"):
        valores = [age, sex, chest_pain, blood_pressure, serum_cholestoral, fasting_blood_sugar,
                   electrocardiographic, max_heart_rate, induced_angina, ST_depression, slope,
                   no_of_vessels, thal]
        # El dato nuevo se estandariza con el mismo escalador del entrenamiento
        x = escalador.transform(pd.DataFrame([valores], columns=VARIABLES))
        probs = {n: probabilidad(fichas[n]["formato"], modelos[n], x) for n in nombres}

        prob = probs[elegido]
        st.metric(f"Probabilidad de enfermedad cardíaca ({elegido})", f"{prob:.1%}")
        st.progress(float(min(max(prob, 0.0), 1.0)))
        if prob >= 0.5:
            st.error(f"Predicción de {elegido}: **ENFERMO (1)**. Se recomienda evaluación médica.")
        else:
            st.success(f"Predicción de {elegido}: **SANO (0)**.")

        st.write("**Resultado de los cuatro algoritmos para este paciente:**")
        resumen = pd.DataFrame({
            "Tipo": {n: fichas[n]["tipo_algoritmo"] for n in nombres},
            "Probabilidad de enfermo": {n: f"{probs[n]:.1%}" for n in nombres},
            "Predicción": {n: "Enfermo (1)" if probs[n] >= 0.5 else "Sano (0)" for n in nombres},
        })
        resumen.index.name = "Algoritmo"
        st.dataframe(resumen)

# ---- Pestaña 2: comparativa ----
with tab_comp:
    st.subheader("Comparativa de machine learning y deep learning")
    if os.path.exists(ruta("comparativa_examen.csv")):
        tabla = pd.read_csv(ruta("comparativa_examen.csv"))
        for conjunto in ["Entrenamiento", "Prueba"]:
            st.write(f"**{conjunto}**")
            st.dataframe(tabla[tabla["Conjunto"] == conjunto].drop(columns="Conjunto").set_index("Algoritmo"))
    for archivo, titulo in [("roc_curvas.png", "Curvas ROC de cada algoritmo"),
                            ("curvas_aprendizaje.png", "Curvas de rendimiento y pérdida de la red neuronal"),
                            ("matrices_confusion.png", "Matrices de confusión en prueba")]:
        if os.path.exists(ruta(archivo)):
            st.write(f"**{titulo}**")
            st.image(ruta(archivo))

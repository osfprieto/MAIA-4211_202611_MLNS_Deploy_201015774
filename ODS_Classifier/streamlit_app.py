from pathlib import Path

import joblib
import streamlit as st

ODS_NAMES = {
	1: "Fin de la pobreza",
	2: "Hambre cero",
	3: "Salud y bienestar",
	4: "Educación de calidad",
	5: "Igualdad de género",
	6: "Agua limpia y saneamiento",
	7: "Energía asequible y no contaminante",
	8: "Trabajo decente y crecimiento económico",
	9: "Industria, innovación e infraestructura",
	10: "Reducción de las desigualdades",
	11: "Ciudades y comunidades sostenibles",
	12: "Producción y consumo responsables",
	13: "Acción por el clima",
	14: "Vida submarina",
	15: "Vida de ecosistemas terrestres",
	16: "Paz, justicia e instituciones sólidas",
	17: "Alianzas para lograr los objetivos",
}

MODEL_PATH = Path(__file__).parent / "resources" / "models" / "ods_classifier.joblib"


@st.cache_resource
def load_model():
	return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Clasificador ODS")
st.title("Clasificador de ODS")
st.write("Desarrollado por Omar Prieto y Jhon Gómez")
st.write("Aplicación y modelo disponible en https://github.com/osfprieto/MAIA-4211_202611_MLNS_Deploy_201015774")

with st.form("classification_form"):
	text = st.text_area(
		"Texto para clasificar",
		placeholder="Escribe aquí el texto que deseas clasificar...",
		height=180,
	)
	classify = st.form_submit_button("Clasificar", type="primary")

if classify:
	if not text.strip():
		st.warning("Escribe un texto antes de clasificar.")
	else:
		try:
			model = load_model()
			ods_number = int(model.predict([text.strip()])[0])
			ods_name = ODS_NAMES.get(ods_number)

			if ods_name is None:
				st.error(f"El modelo devolvió una clase desconocida: {ods_number}.")
			else:
				st.success(f"ODS {ods_number}: {ods_name}")
		except Exception as error:
			st.error(f"No se pudo clasificar el texto: {error}")

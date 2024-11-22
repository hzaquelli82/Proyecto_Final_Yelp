#Importamos las librerías necesarias 

import streamlit as st
import pandas as pd
import numpy as np
# import google
import os
# import re
from bertopic import BERTopic
# import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from google.oauth2 import service_account
from google.oauth2.service_account import Credentials
from google.cloud import bigquery
from PIL import Image

import nltk
# import os

nltk_data_dir = "./resources/nltk_data_dir/"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
# Descargar stopwords si no lo has hecho antes
# nltk.download('stopwords')

# Obtener la lista de stopwords en español
custom_stopwords = stopwords.words('english')

from sklearn.feature_extraction.text import CountVectorizer

# Configurar TOKENIZERS_PARALLELISM
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Crear el modelo de BERTopic
modelo_bertopic = BERTopic(n_gram_range=(1, 2), language='english', vectorizer_model = CountVectorizer(stop_words=custom_stopwords))


# CREAR CONEXION Y DESCARGAR TABLA DE BIGQUERY 
# CREDENTIALS_PATH = '.streamlit/secrets.toml'
PROJECT_ID = "proyecto-final-439222"  

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

# Especificar la tabla
DATASET_ID = "ds_proyecto_nordsee"  # Reemplaza con el ID de tu dataset
TABLE_ID = "tabla_modelo"
PORCENTAJE = 10

# Crear la consulta
query = f"""
SELECT *
FROM {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}
TABLESAMPLE SYSTEM ({PORCENTAJE} PERCENT)
"""
# Ejecutar la consulta y obtener resultados como DataFrame
query_job = client.query(query)
results = query_job.result()
df = results.to_dataframe()

# Convertir la columna 'time' a datetime, especificando el formato
df['time'] = pd.to_datetime(df['time'])


imagen = Image.open("logo.jpeg")

with st.sidebar:
    st.image(imagen, use_container_width=True)

    st.title('Filtros')
        
    años_disponibles = sorted(df["time"].dt.year.unique())
    años_disponibles = [año for año in años_disponibles if año >= 2018]

    # Crear selectores para mes y año
    year = st.sidebar.selectbox('Selecciona un año', años_disponibles)
    month = st.sidebar.selectbox('Selecciona un mes', range(1, 13), format_func=lambda x: f"Mes {x}")
    stars = st.sidebar.selectbox('Selecciona una puntuación', range(1, 6), format_func=lambda x: f"{x} estrellas")
    state = st.sidebar.selectbox('Selecciona un Estado', sorted(df["id_state"].unique()))
    modelo = st.button('Iniciar Análisis')

# Filtrar los datos según la selección del usuario
filtered_data = df[(df["time"].dt.year == year) & (df["time"].dt.month == month) & (df["stars_review"] == stars) & (df["id_state"] == state)]

# Función para graficar los 10 tópicos más frecuentes
def plot_top_topics(df_topics):
    """Grafica los 10 tópicos más frecuentes.

    Args:
        df_topics (pd.DataFrame): DataFrame con la información de los tópicos.
    """
    top_10_topics = df_topics.nlargest(10, 'Count')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Count', y='Name', data=top_10_topics, palette=['#00008B', '#FF0000'])
    plt.xlabel('Frecuencia')
    plt.ylabel('Tópico')
    plt.title('Los 10 Tópicos Más Frecuentes')
    st.pyplot(plt)

# Función para entrenar el modelo
def entrenar_modelo(df):
    documentos = df["review_text"]

    # Entrenar el modelo con los documentos
    temas, probabilidad = modelo_bertopic.fit_transform(documentos)

    df_topics = modelo_bertopic.get_topic_info()

    # Reemplazar guiones bajos por espacios y eliminar números
    df_topics['Name'] = df_topics['Name'].str.replace('_', ' ').str.replace('\d+', '')

    # Eliminar espacios en blanco adicionales al inicio y al final
    df_topics['Name'] = df_topics['Name'].str.replace(r'\d+', '', regex=True).str.strip()

    return df_topics

# Función para mostrar la nube de palabras en Streamlit
def show_wordcloud(wordcloud):
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)  # Ajustar el layout para evitar recortes

    # Mostrar la figura en Streamlit
    st.pyplot(plt)


st.title('Análisis de sentimientos en las reseñas de Yelp y Maps')

st.subheader('Principales tópicos según año, mes, puntuación en estrellas y estado seleccionado.')

col0 = st.columns((1, 4, 1), gap='medium')

with col0[1]:   
    if modelo:  # Ejecutar el análisis solo si se presiona el botón
        # Entrenar el modelo cuando se presiona el botón
        df_topics = entrenar_modelo(filtered_data)
        plot_top_topics(df_topics)

        # Crear listas de tópicos para armar nube de palabras
        tema_1_palabras = modelo_bertopic.get_topic(-1)
        tema_2_palabras = modelo_bertopic.get_topic(0)
        tema_3_palabras = modelo_bertopic.get_topic(1)
        tema_4_palabras = modelo_bertopic.get_topic(2)

        # Convertir la lista a un diccionario para cada tópico
        word_dict_1 = dict(tema_1_palabras)
        word_dict_2 = dict(tema_2_palabras)
        word_dict_3 = dict(tema_3_palabras)
        word_dict_4 = dict(tema_4_palabras)

        col1 = st.columns((1, 1), gap='medium')

        with col1[0]:
            wordcloud = WordCloud(width=800, height=400, background_color='#7fb3d5').generate_from_frequencies(word_dict_1)
            # Llamar a la función para mostrar la nube de palabras
            show_wordcloud(wordcloud)

        with col1[1]:
            wordcloud1 = WordCloud(width=800, height=400, background_color='#a9cce3').generate_from_frequencies(word_dict_2)
            # Llamar a la función para mostrar la nube de palabras
            show_wordcloud(wordcloud1)

        col2 = st.columns((1, 1), gap='medium')

        with col2[0]:
            wordcloud2 = WordCloud(width=800, height=400, background_color='#d4e6f1').generate_from_frequencies(word_dict_3)
            # Llamar a la función para mostrar la nube de palabras
            show_wordcloud(wordcloud2)

        with col2[1]:
            wordcloud3 = WordCloud(width=800, height=400, background_color='#eaf2f8').generate_from_frequencies(word_dict_4)
            # Llamar a la función para mostrar la nube de palabras
            show_wordcloud(wordcloud3)

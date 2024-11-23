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


# Descargar stopwords si no lo has hecho antes
nltk.download('stopwords')

# Obtener la lista de stopwords en español
custom_stopwords = stopwords.words('english')

from sklearn.feature_extraction.text import CountVectorizer

# Configurar TOKENIZERS_PARALLELISM
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Crear el modelo de BERTopic
modelo_bertopic = BERTopic(n_gram_range=(1, 2), language='english', vectorizer_model = CountVectorizer(stop_words=custom_stopwords))


# credenciales 
# CREDENTIALS_PATH = '.streamlit/secrets.toml'
PROJECT_ID = "proyecto-final-439222"  

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

st.title('Análisis de sentimentos en las reseñas de Yelp y Maps')

logo = 'logo.jpeg'

# Muestra la imagen en la barra lateral con un estilo personalizado
st.sidebar.image(logo, use_column_width=True)  # Ocupa todo el ancho de la columna

st.sidebar.title('Filtros')
    
años_disponibles = sorted(df["time"].dt.year.unique())
años_disponibles = [año for año in años_disponibles if año >= 2018]

# Crear selectores para mes y año
year = st.sidebar.selectbox('Selecciona un año', años_disponibles)
month = st.sidebar.selectbox('Selecciona un mes', range(1, 13))
stars= st.sidebar.selectbox('Selecciona una puntuación', range(1, 6), format_func=lambda x: f"{x} estrellas")
state= st.sidebar.selectbox('Selecciona un Estado', sorted(df["id_state"].unique()))


# Filtrar los datos según la selección del usuario
filtered_data = df[(df["time"].dt.year == year) & (df["time"].dt.month == month)]
#st.write(filtered_data)
st.subheader('Top 10 tópicos más frecuentes según parámetros seleccionados')

documentos= filtered_data["review_text"]


# Entrenar el modelo con los documentos
temas, probabilidad = modelo_bertopic.fit_transform(documentos)

df_topics= modelo_bertopic.get_topic_info()

# Reemplazar guiones bajos por espacios y eliminar números
df_topics['Name'] = df_topics['Name'].str.replace('_', ' ').str.replace('\d+', '')

# Eliminar espacios en blanco adicionales al inicio y al final
df_topics['Name'] = df_topics['Name'].str.replace(r'\d+', '', regex=True).str.strip()

#st.write(df_topics)

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


plot_top_topics(df_topics)

#plt.figtext(0.5, 0.95, "Palabras más frecuentes, por orden de importancia", ha='center', fontsize=16, fontweight='bold', color='darkblue')
st.subheader("Palabras más frecuentes, por orden de importancia")

tema_1_palabras = modelo_bertopic.get_topic(-1)

# Convertir la lista a un diccionario
word_dict_1 = dict(tema_1_palabras)

color_palette = sns.color_palette("coolwarm", as_cmap=True)

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=color_palette).generate_from_frequencies(word_dict_1)
# Crear la nube de palabras
#wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_dict_1)

# Función para mostrar la nube de palabras en Streamlit
def show_wordcloud(wordcloud):
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)  # Ajustar el layout para evitar recortes

    # Mostrar la figura en Streamlit
    st.pyplot(plt)

# Llamar a la función para mostrar la nube de palabras
show_wordcloud(wordcloud)



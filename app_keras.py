import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def carrega_modelo():

    # URL do modelo .keras
    # https://drive.google.com/file/d/1TjXwqaXmlmhildNXq2BUgwVTgaDJphkR/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1TjXwqaXmlmhildNXq2BUgwVTgaDJphkR'
    
    # Nome do arquivo a ser salvo
    nome = 'modelo.keras'

    # Faz o download do modelo
    gdown.download(url, nome, quiet=False)

    # Carrega o modelo .keras
    modelo = load_model(filepath=nome)
    
    st.write(f'Modelo carregado: {nome}')

    return modelo

def carrega_imagem():
    
    # Cria um file uploader que permite o usuário carregar imagens
    uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        # Para ler a imagem como um objeto PIL Image
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        # Mostrar a imagem carregada
        st.image(image)
        st.success("Imagem carregada com sucesso!")

        # Pré-processamento da imagem
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalização para o intervalo [0, 1]
        image = np.expand_dims(image, axis=0)

        return image

def previsao(modelo, image):

    # Executa a inferência com o modelo .keras
    output_data = modelo.predict(image)

    classes = ['Tombada', 'Em_pe']

    # Converte as probabilidades para porcentagens
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]
    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
                 title='Probabilidade de haver lata tombada')
    st.plotly_chart(fig)

def main():

    st.set_page_config(
        page_title="Identifica latas tombadas",
        page_icon="🧠",
    )
    
    st.write("# Classifica latas tombadas")
    

    modelo = carrega_modelo()

    image = carrega_imagem()

    if image is not None:
        previsao(modelo, image)
    
if __name__ == "__main__":
    main()

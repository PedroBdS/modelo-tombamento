import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import keras

@st.cache_resource
def carrega_modelo():
    # url = 'https://drive.google.com/uc?id=1-4_L_9H1Xd5v0OSR3KK5SAkGGbYuAOlx'
    
    # gdown.download(url,'modelo_quantizado16bits.tflite')
    # interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    # interpreter.allocate_tensors()
    
    # return interpreter

    # https://drive.google.com/file/d/1XwTwZAMhevW4X_z2c0fKJFW6Rf1VkIWT/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1XwTwZAMhevW4X_z2c0fKJFW6Rf1VkIWT'
    
    gdown.download(url, 'classificacao_de_latas.h5')
    interpreter = keras.models.load_model('classificacao_de_latas.h5')
    return interpreter

    # modelo = st.file_uploader("Arraste e solte a classificacao_de_latas.h5 aqui ou clique para selecionar", type=['h5'])
    # if modelo is not None:
    #     interpreter = keras.models.load_model("classificacao_de_latas.h5")
    #     return interpreter

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

        #Pré-processamento da imagem
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalização para o intervalo [0, 1]
        image = np.expand_dims(image, axis=0)

        return image

def previsao(interpreter,image):
# Realiza a previsão diretamente com o modelo Keras
    output_data = interpreter.predict(image)

    # Classes de saída (ajustar conforme necessário)
    classes = ['tombada', 'empe']

    # Formata os resultados em um DataFrame
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]
    
    # Cria o gráfico com Plotly
    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
                 title='Probabilidade de haver lata tombada')
    
    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig)


    '''
    # Obtém detalhes dos tensores de entrada e saída
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define o tensor de entrada para o modelo
    interpreter.set_tensor(input_details[0]['index'], image)

    # Executa a inferência
    interpreter.invoke()

    # Obtém a saída do modelo
    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['tombada', 'empe']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]
    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
             title='Probabilidade de haver lata tombada')
    st.plotly_chart(fig)
    '''
def main():

    st.set_page_config(
        page_title="Identifica latas tombadas",
        page_icon="🍇",
    )
    
    st.write("# Classifica latas tombadas")
    

    interpreter = carrega_modelo()

    image = carrega_imagem()

    if image is not None:

        previsao(interpreter,image)
    


if __name__ == "__main__":
    main()
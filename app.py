import pandas as pd
import numpy as np
import pickle
import streamlit as st
import time

pickle_in = open('knn_model.pkl', 'rb')
knn = pickle.load(pickle_in)

st.set_page_config(page_title="SpecieScanner Apps")

def prediction(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
    prediction = knn.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    print(prediction)
    return prediction

def main():
    st.title("SpecieScanner Apps")
    st.header("Pemindai Spesies Bunga Iris Menggunakan Algoritma K-Nearest Neighbour (KNN)")
    
    st.subheader("By: Agung Gunawan")
    st.write('\n')
    
    st.image("images/irisflowers.png", width=700)
    st.write('\n')
    st.write('\n')
    
    st.sidebar.markdown("<strong>Sebelum memindai, isi data berikut :</strong>", 
                        unsafe_allow_html=True)
    SepalLengthCm = st.sidebar.number_input(label="Sepal Length (cm)", step=1., format="%.1f")
    SepalWidthCm = st.sidebar.number_input(label="Sepal Width (cm)", step=1., format="%.1f")
    PetalLengthCm = st.sidebar.number_input(label="Petal Length (cm)", step=1., format="%.1f")
    PetalWidthCm = st.sidebar.number_input(label="Petal Width (cm)", step=1., format="%.1f")
    result =""
    
    st.write('\n')
    if st.button("PINDAI SPESIES"):
        result = prediction(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
        for result_predict in result:
            print(result_predict)
            
        if result_predict == 'Iris-setosa':
            result_predict = 'Iris Setosa'
        elif result_predict == 'Iris-versicolor':
            result_predict = 'Iris Versicolor'
        else:
            result_predict = 'Iris Virginica'
        
        st.write('\n')
        st.write('\n')
        st.write('\n')
        with st.spinner('Sedang Memindai...'):
            time.sleep(2)
        st.success(f'Sukses, berdasarkan sepal dan petal, iris termasuk spesies {result_predict}.')
        
        if result_predict == "Iris Setosa":
            st.image("images/iris-setosa.png", width=350)
        elif result_predict == "Iris Versicolor":
            st.image("images/iris-versicolor.png", width=350)
        else:
            st.image("images/iris-virginica.png", width=350)
            
if __name__=='__main__':
    main()
    
    
#menghilangkan burger dan made with streamlit
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

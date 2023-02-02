import pandas as pd
import numpy as np
import pickle
import streamlit as st
import time

pickle_in = open('knn_model.pkl', 'rb')
knn = pickle.load(pickle_in)

st.set_page_config(page_title="SpecieScanner Apps")

@st.cache(suppress_st_warning = True)

def prediction(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
    prediction = knn.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    print(prediction)
    return prediction

def main():
    st.sidebar.title("SpecieScanner Apps")
    st.sidebar.write('\n')
        
    app_menu = st.sidebar.selectbox('MENU', ['Home', 'Scan'])
    
    if app_menu == 'Home':
        st.title("SpecieScan Apps")
        st.markdown("Selamat Datang.")
        
        #st.image('assets/loan_image.jpg')
        st.write('\n')
        st.write('\n')
        
        st.subheader('Dataset Iris:')
        data = pd.read_csv('iris.csv')
        
        if st.checkbox("Tampilkan/Sembunyikan Seluruh Data"):
            st.write(data)
        
        if st.checkbox("Tampilkan/Sembunyikan hanya 10 Data Teratas"):
            st.write(data.head(10))
        
        if st.checkbox("Tampilkan/Sembunyikan hanya 10 Data Terakhir"):
            st.write(data.tail(10))
        
        st.write('\n')
        st.write('\n')
        st.subheader('Sepal Length, SepalWidth, Petal Length, dan Petal Width vs Species :')
        
        if st.checkbox("Tampilkan/Sembunyikan Line Chart") :
            st.line_chart( data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','Species']].head(10) )
        
    else :
        st.header("Pemindai Spesies Bunga Iris Menggunakan Algoritma K-Nearest Neighbour (KNN)")
        st.markdown("By: Agung Gunawan | https://github.com/insomniagung/")
        st.write('\n')
        st.image("images/irisflowers.png", width=700)
        st.write('\n')
        st.write('\n')
        
        st.sidebar.write('\n')
        st.sidebar.write('\n')
        st.sidebar.markdown("<strong>Sebelum memindai, isi data berikut :</strong>", unsafe_allow_html=True)
    
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

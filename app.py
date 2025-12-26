"""
Created on Fri Dec 26 21:48:39 2025

@author: thaer
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1. Başlık ve Yapılandırma [cite: 86, 110]
st.set_page_config(page_title="MIS Projesi", layout="centered")
st.title("📊 Veri Analizi ve Görselleştirme Uygulaması")
st.markdown("Bu uygulama, dönemsel MIS konularını kapsayan bir Streamlit projesidir. [cite: 103]")

# 2. Kenar Çubuğu (Sidebar) [cite: 180, 181]
st.sidebar.header("Ayarlar ve Filtreler")
menu = st.sidebar.selectbox("Sayfa Seçiniz:", ["Ana Sayfa", "Analiz Paneli"]) # [cite: 156]

# 3. Veri Yükleme Bileşeni [cite: 167, 169]
st.sidebar.subheader("Veri Kaynağı")
uploaded_file = st.sidebar.file_uploader("Bir CSV dosyası yükleyin", type="csv")

if uploaded_file is not None:
    # Veriyi Oku [cite: 171]
    df = pd.read_csv(uploaded_file)
    
    if menu == "Ana Sayfa":
        st.header("Veri Setine Genel Bakış")
        st.write("Verinin ilk 5 satırı: [cite: 91]")
        st.dataframe(df.head()) # İnteraktif tablo [cite: 119, 123]
        
        # İstatistiksel Bilgiler [cite: 130]
        st.subheader("Veri İstatistikleri")
        st.write(df.describe())

    elif menu == "Analiz Paneli":
        st.header("İnteraktif Grafik Paneli")
        
        # Kullanıcı etkileşimi: Slider [cite: 149, 150]
        limit = st.slider("Görselleştirilecek veri miktarını seçin:", 5, len(df), 20)
        
        # Grafik Çizimi [cite: 200, 201]
        st.subheader(f"İlk {limit} Kayıt İçin Grafik")
        fig, ax = plt.subplots()
        df.iloc[:limit].plot(kind='bar', ax=ax)
        st.pyplot(fig)
        
        # Başarı mesajı [cite: 207, 211]
        st.success("Grafik başarıyla oluşturuldu!")
else:
    st.info("Lütfen sol taraftaki menüden bir CSV dosyası yükleyerek başlayın.")
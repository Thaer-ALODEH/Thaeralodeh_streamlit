import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sayfa Ayarları
st.set_page_config(page_title="Online Retail Analysis", layout="wide")

st.title("YBSB3003 - Programming for Data Science")
st.header("Streamlit App Exercises - Online Retail")

# Veri Yükleme Fonksiyonu
@st.cache_data
def load_data():
    try:
        # Not: Veri setinizin adının 'online_retail_data.csv' olduğundan emin olun
        df = pd.read_csv("online_retail_data.csv", encoding="ISO-8859-1")
        # Problem 6: Revenue (Gelir) değişkeni oluşturma
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None

df = load_data()

if df is not None:
    # Kenar Çubuğu Navigasyonu
    st.sidebar.title("Ödev Soruları")
    menu = st.sidebar.selectbox("Bir Bölüm Seçin:", 
        ["Ana Dashboard (P12)", "Veri Yapısı (P1-P2)", "Görselleştirmeler (P3-7)", "ML ve PCA (P8-11)"])

    # PROBLEM 1 & 2: Veri Yapısı
    if menu == "Veri Yapısı (P1-P2)":
        st.subheader("1. Veri Setinin İlk 10 Satırı")
        st.dataframe(df.head(10))

        st.subheader("2. Yapısal Bilgiler")
        st.write(f"Gözlem Sayısı: {df.shape[0]}")
        st.write(f"Değişken Sayısı: {df.shape[1]}")
        st.write("Sütun Veri Tipleri:")
        st.write(df.dtypes.astype(str))

    # PROBLEM 3-7: Görselleştirmeler
    elif menu == "Görselleştirmeler (P3-7)":
        # P3: Kategorik Değişken Pasta Grafiği
        st.subheader("3. Kategorik Değişken Dağılımı")
        cat_vars = df.select_dtypes(include=['object']).columns.tolist()
        selected_cat = st.sidebar.selectbox("Kategorik Değişken Seçin:", cat_vars)
        fig3, ax3 = plt.subplots()
        df[selected_cat].value_counts().head(10).plot.pie(autopct='%1.1f%%', ax=ax3)
        st.pyplot(fig3)

        # P4 & P5: Bar ve Scatter
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("4. Ülkelere Göre İşlemler (Top 10)")
            st.bar_chart(df['Country'].value_counts().head(10))
        with col_b:
            st.subheader("5. Miktar vs Birim Fiyat")
            fig5, ax5 = plt.subplots()
            sns.scatterplot(data=df, x='Quantity', y='UnitPrice', ax=ax5)
            st.pyplot(fig5)

        # P6 & P7: Histogram ve Zaman Serisi
        st.subheader("6. Revenue (Gelir) Dağılımı")
        fig6, ax6 = plt.subplots()
        sns.histplot(df['Revenue'], bins=50, kde=True, ax=ax6)
        st.pyplot(fig6)

        st.subheader("7. Zaman İçinde İşlem Sayısı")
        df_time = df.set_index('InvoiceDate').resample('M').size()
        st.line_chart(df_time)

    # PROBLEM 8-11: İleri Analiz
    elif menu == "ML ve PCA (P8-11)":
        numeric_df = df[['Quantity', 'UnitPrice', 'Revenue']].dropna()
        
        # P8-P9: PCA
        st.subheader("8 & 9. PCA Analizi (2 Bileşen)")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(scaled_data)
        
        pca_df = pd.DataFrame(pca_res, columns=['PCA1', 'PCA2'])
        pca_df['Country'] = df['Country']
        top_5 = df['Country'].value_counts().head(5).index
        pca_filtered = pca_df[pca_df['Country'].isin(top_5)]
        
        fig8, ax8 = plt.subplots()
        sns.scatterplot(data=pca_filtered, x='PCA1', y='PCA2', hue='Country', ax=ax8)
        st.pyplot(fig8)

        # P10-P11: Random Forest
        st.subheader("10 & 11. Random Forest ile Revenue Tahmini")
        if st.button("Modeli Eğit"):
            X = numeric_df[['Quantity', 'UnitPrice']]
            y = numeric_df['Revenue']
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            st.write("Önemli Özellikler (Feature Importance):")
            st.bar_chart(pd.Series(rf.feature_importances_, index=X.columns))
            
            mse = mean_squared_error(y, rf.predict(X))
            st.metric("Model Performansı (MSE)", f"{mse:.2f}")

    # PROBLEM 12: Yönetici Dashboard
    elif menu == "Ana Dashboard (P12)":
        st.header("Yönetici Karar Destek Paneli")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Ülke Bazlı Gelir")
            st.bar_chart(df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10))
        with c2:
            st.subheader("PCA İşlem Desenleri")
            # PCA görseli buraya da eklenebilir
            st.write("PCA analizi, işlemlerin hacim ve değer bazlı kümelendiğini göstermektedir.")

        st.info("**Yönetsel Sorular:** En çok gelir Birleşik Krallık'tan sağlanmaktadır. Bu dashboard, stok yönetimi ve bölge bazlı pazarlama stratejileri için kullanılabilir.")

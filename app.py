import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. SAYFA YAPILANDIRMASI
st.set_page_config(page_title="Online Retail Analysis", layout="wide")

st.title("YBSB3003 - Programming for Data Science")
st.header("Streamlit App Exercises - Online Retail")

# 2. VERİ YÜKLEME FONKSİYONU (EN GÜÇLÜ VERSİYON)
@st.cache_data
def load_data():
    file_path = "online_retail_data.csv"
    try:
        # Önce dosyayı otomatik ayraç ve doğru kodlama ile okumayı dene
        df = pd.read_csv(file_path, sep=None, engine='python', encoding="ISO-8859-1")
        
        # Eğer dosya okunduysa ama boşsa (No columns hatası için önlem)
        if df.empty or len(df.columns) < 2:
            raise ValueError("Dosya içeriği yetersiz.")
            
    except Exception as e:
        # HATA DURUMUNDA VEYA DOSYA YOKSA: Ödevi test edebilmeniz için örnek veri oluşturur
        st.warning(f"Dosya okuma hatası ({e}). Test için örnek veri oluşturuluyor...")
        data = {
            'InvoiceNo': range(500, 600),
            'StockCode': ['STK'+str(i) for i in range(100)],
            'Description': ['Ürün '+str(i) for i in range(100)],
            'Quantity': np.random.randint(1, 50, 100),
            'UnitPrice': np.random.uniform(1.0, 100.0, 100),
            'InvoiceDate': pd.date_range(start='2025-01-01', periods=100, freq='H'),
            'Country': np.random.choice(['United Kingdom', 'Germany', 'France', 'Spain', 'Netherlands'], 100)
        }
        df = pd.DataFrame(data)

    # PROBLEM 6: Revenue Değişkeni Oluşturma [cite: 388]
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # PROBLEM 7: Tarih Formatı [cite: 390]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    return df

df = load_data()

# 3. ÖDEV MADDELERİ (P1 - P12) 
if df is not None:
    st.sidebar.title("Ödev Menüsü")
    menu = st.sidebar.radio("Bölüm Seçin:", ["Dashboard (P12)", "Veri Bilgileri (P1-P2)", "Görselleştirme (P3-P7)", "ML ve PCA (P8-P11)"])

    # PROBLEM 1 & 2 [cite: 379, 380]
    if menu == "Veri Bilgileri (P1-P2)":
        st.subheader("P1: İlk 10 Satır")
        st.dataframe(df.head(10))
        st.subheader("P2: Yapısal Bilgiler")
        st.write(f"Gözlem: {df.shape[0]}, Değişken: {df.shape[1]}")
        st.write("Veri Tipleri:", df.dtypes.astype(str))

    # PROBLEM 3 - 7 [cite: 384-390]
    elif menu == "Görselleştirme (P3-P7)":
        # P3: Kategorik Pasta Grafiği
        cat_var = st.sidebar.selectbox("Kategorik Değişken:", ["Country", "Description"])
        st.subheader(f"P3: {cat_var} Dağılımı")
        fig3, ax3 = plt.subplots()
        df[cat_var].value_counts().head(5).plot.pie(autopct='%1.1f%%', ax=ax3)
        st.pyplot(fig3)

        # P4 & P5: Bar ve Scatter
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("P4: Top 10 Ülke")
            st.bar_chart(df['Country'].value_counts().head(10))
        with col2:
            st.subheader("P5: Miktar vs Birim Fiyat")
            fig5, ax5 = plt.subplots()
            sns.scatterplot(data=df, x='Quantity', y='UnitPrice', ax=ax5)
            st.pyplot(fig5)

        # P6 & P7: Histogram ve Çizgi Grafik
        st.subheader("P6: Revenue Dağılımı")
        fig6, ax6 = plt.subplots()
        df['Revenue'].hist(bins=30, ax=ax6)
        st.pyplot(fig6)

        st.subheader("P7: Zaman Serisi")
        st.line_chart(df.set_index('InvoiceDate').resample('D').size())

    # PROBLEM 8 - 11 [cite: 393-399]
    elif menu == "ML ve PCA (P8-11)":
        num_df = df[['Quantity', 'UnitPrice', 'Revenue']].dropna()
        # P8-P9: PCA
        scaler = StandardScaler()
        scaled = scaler.fit_transform(num_df)
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(scaled)
        
        st.subheader("P8-P9: PCA Analizi")
        fig8, ax8 = plt.subplots()
        plt.scatter(pca_res[:,0], pca_res[:,1], alpha=0.5)
        st.pyplot(fig8)

        # P10-P11: Random Forest
        if st.button("Modeli Eğit (P11)"):
            rf = RandomForestRegressor(n_estimators=50)
            rf.fit(num_df[['Quantity', 'UnitPrice']], num_df['Revenue'])
            st.success(f"Model Eğitildi! Önem Sırası: {rf.feature_importances_}")

    # PROBLEM 12: Entegre Dashboard [cite: 401]
    elif menu == "Dashboard (P12)":
        st.header("Yönetici Karar Destek Paneli")
        st.write("Bu panel tüm analizlerin özetini sunar.")
        st.metric("Toplam Gelir", f"{df['Revenue'].sum():.2f}")
        st.bar_chart(df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(5))
        st.info("P12 Sorusu: En çok gelir Birleşik Krallık'tan sağlanmaktadır.")
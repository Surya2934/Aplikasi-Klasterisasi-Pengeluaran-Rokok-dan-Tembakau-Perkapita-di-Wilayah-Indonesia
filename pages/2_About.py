import streamlit as st

st.set_page_config(
    page_title="Klasterisasi Pola Rokok",
    page_icon="🚬",
    layout="wide")

with st.sidebar:
    st.title("Dashboard Klasterisasi")

st.title("Tentang Aplikasi Dashboard Klasterisasi")

st.info("""
    Aplikasi ini merupakan prototype yang dirancang sebagai bagian dari pembuatan skripsi untuk memenuhi
    persyaratan kelulusan program studi Teknik Informatika di Universitas Tarumanagara.
""")

st.header("Tujuan")
st.write("""
    Tujuan utama dari perancangan ini adalah untuk menyediakan sebuah alat bantu analisis
    yang dapat mempermudah pengguna dalam mengidentifikasi dan memvisualisasikan
    pola-pola pengeluaran rokok dan tembakau di tingkat kabupaten/kota di Indonesia
    menggunakan metode klasterisasi K-Means dan K-Means++ atau OPTICS.
""")

st.header("Teknologi yang Digunakan")
st.markdown("""
- **Bahasa Pemrograman:** Python
- **Framework Aplikasi Web:** Streamlit
- **Library Analisis Data:** Pandas, Scikit-learn
- **Library Visualisasi:** Seaborn, Matplotlib, Geopandas
""")

st.header("Disusun Oleh")
st.markdown("""
- **Nama:** Surya Dharma Kang
- **NIM:** 535220148
""")
import streamlit as st
import pandas as pd
import io

st.set_page_config(
    page_title="Klasterisasi Pola Rokok",
    page_icon="🚬",
    layout="wide"
)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def to_excel(df):
    output = io.BytesIO()
    # Menggunakan openpyxl sebagai engine untuk menulis ke file Excel
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Template')
    processed_data = output.getvalue()
    return processed_data

with st.sidebar:
    st.title("Dashboard Klasterisasi")

st.title("Clustering Pola Pengeluaran Rokok dan Tembakau di Indonesia")
st.info("""Selamat datang di Dashboard Analisis Pola Pengeluaran Rokok dan Tembakau.
        Aplikasi ini dirancang untuk membantu Anda memahami keragaman pola konsumsi rokok di seluruh kabupaten/kota di Indonesia melalui pendekatan klasterisasi K-Means.
        """)

col1, col2, col3 = st.columns([3,1,1])
with col1:
    st.image("assets/foto_rokok.jpg", width=500)

with col2:
    st.write("`Dataset sudah tersedia`") 
    st.write("Syarat Dataset yang digunakan jika ingin memakai dataset pribadi: " 
    "\n 1. File bertipe .csv / .xlxs" 
    "\n 2. File berisi seluruh kabupaten/kota di Indonesia" 
    "\n 3. Tidak ada duplikat"
    "\n 4. Kolom pada file sesuai dengan template"
    "\n (Kolom `ROKOK DAN TEMBAKAU` adalah nilai total dari seluruh jenis rokok yang ada)")

with col3:
    st.write("Template File Dataset")

    template_df = pd.DataFrame({
        'Kabupaten/Kota': ["Contoh: Kota Bandung"],
        'Tahun': [2024],
        'ROKOK DAN TEMBAKAU': [24000.0],
        'Rokok kretek filter': [15000.0],
        'Rokok kretek tanpa filter': [5000.0],
        'Rokok putih': [2500.0],
        'Tembakau': [1000.0],
        'Rokok dan tembakau Lainnya': [500.0]
    })

    excel_template = to_excel(template_df)

    st.download_button(
    label="📊 Unduh Template Dataset (.xlsx) ",
    data=excel_template,
    file_name='template_data_rokok.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

    manual_book_url = "https://drive.google.com/drive/folders/1ljWgVD--7vg_3oj7gc9Uc-Ka4NBMYLtI?usp=sharing"
    
    st.write("Manual Book")
    st.markdown(f"""
    <a href="{manual_book_url}" target="_blank" style="text-decoration: none;">
        <button style="
            width: 100%;
            padding: 0.3rem;
            font-size: 1rem;
            font-weight: bold;
            color: white;
            background-color: #000000;
            border: 2px solid #FFFFFF;
            border-radius: 0.5rem;
            cursor: pointer;
        ">
            Buka Manual Book (.pdf)
        </button>
    </a>
    """, unsafe_allow_html=True)

st.markdown("---")
st.header("TUTORIAL PEMAKAIAN WEB 📖")
st.write("🌟 Lihat `MANUAL BOOK` untuk memahami lebih dalam mengenai website dan pengertian setiap atribut pada web 🌟")
st.write("1. Pergi ke halaman `Klasterisasi` untuk memulai proses klasterisasi")
st.write("2. Input Dataset atau gunakan Dataset bawaan bila tidak memiliki Dataset")
st.write("3. Input parameter yang tersedia sesuai dengan keingingan anda pada area `Pengaturan Analisis`")
st.image("assets/step2.png", width=600)

st.markdown("---")
st.subheader("Untuk K-Means dan K-Means++")
st.write("4. Klik tombol `cari rekomendasi nilai K` seperti di gambar untuk mencari jumlah kelompok paling optimal")
st.image("assets/k_means3.png", width=200)
st.write("5. Setelah proses selesai slider akan muncul. Pilih nilai `K` dengan menggeser slider tersebut.")
st.write("6. Klik tombol `Klasterisasi` untuk memulai proses pengelompokan dan hasil akan muncul")

st.markdown("---")
st.subheader("Untuk OPTICS")
st.write("4. Tentukan jumlah `sampel minimum` yang ingin digunakan untuk klasterisasi data")
st.write("5. Klik tombol `Klasterisasi` untuk memulai proses pengelompokan dan hasil akan muncul")
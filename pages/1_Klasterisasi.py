import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import time
import numpy as np
import geopandas as gpd 

# --- Import kode K-Means++ Manual ---
try:
    from k_means_plus_plus import KMeansPlusPlusManual
except ImportError:
    st.error("File 'k_means_plus_plus.py' tidak ditemukan.")
    KMeansPlusPlusManual = None

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Utama", layout="wide")

# --- FUNGSI-FUNGSI UTAMA ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Kabupaten/Kota' in df.columns and 'Tahun' in df.columns:
            df.drop_duplicates(subset=['Kabupaten/Kota', 'Tahun'], keep='first', inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error: Gagal memuat file '{file_path}'. Detail: {e}")
        return None

@st.cache_data
def load_geodata(file_path):
    try:
        gdf = gpd.read_file(file_path)
        return gdf
    except Exception as e:
        st.error(f"Error: Gagal memuat file GeoJSON '{file_path}'. Detail: {e}")
        return None

def run_clustering(df, selected_features, algorithm_choice, params):
    if not selected_features:
        st.error("Harap pilih setidaknya satu fitur.")
        return {}
    
    df_cluster = df.copy()
    df_cluster[selected_features] = df_cluster[selected_features].fillna(0)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_cluster[selected_features])

    model = None
    if algorithm_choice == 'K-Means++':
        if KMeansPlusPlusManual:
            model = KMeansPlusPlusManual(n_clusters=params['k'], random_state=42)
    elif algorithm_choice == 'K-Means':
        model = KMeans(n_clusters=params['k'], init='random', n_init=10, random_state=42)
    elif algorithm_choice == 'OPTICS':
        model = OPTICS(min_samples=params['min_samples'], xi=0.05, min_cluster_size=5)

    if model is None:
        st.error("Gagal menginisialisasi model klasterisasi.")
        return {}
        
    df_cluster['Cluster'] = model.fit_predict(X_scaled)
    
    labels = df_cluster['Cluster']
    results = {'df_result': df_cluster, 'X_scaled': X_scaled, 'model': model}

    non_noise_mask = labels != -1
    labels_non_noise = labels[non_noise_mask]
    X_non_noise = X_scaled[non_noise_mask]
    n_clusters_found = len(set(labels_non_noise))

    if n_clusters_found > 1 and len(X_non_noise) > n_clusters_found:
        try:
            results['sil'] = silhouette_score(X_non_noise, labels_non_noise)
            results['dbi'] = davies_bouldin_score(X_non_noise, labels_non_noise)
        except ValueError:
            pass # Skor tidak dapat dihitung
    
    if algorithm_choice == 'OPTICS':
        results['n_clusters_found'] = n_clusters_found
        results['n_noise'] = np.sum(labels == -1)

    return results

# --- TAMPILAN HALAMAN UTAMA ---
st.title("Halaman Analisis Utama Klasterisasi 🔬")

# --- BAGIAN INPUT DATA (BARU) ---
st.header("Sumber Data")
source_choice = st.radio(
    "Pilih sumber data untuk analisis:",
    ('Gunakan Dataset Bawaan Website', 'Unggah File Sendiri'),
    horizontal=True,
)

if source_choice == 'Unggah File Sendiri':
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx'])
    if uploaded_file:
        df_analysis = load_data(uploaded_file)
        if df_analysis is not None:
            st.session_state['df_analysis'] = df_analysis
            st.success("File Anda berhasil diunggah!")
else:
    df_analysis = load_data('dataset_rokok.csv')
    if df_analysis is not None:
        st.session_state['df_analysis'] = df_analysis

if 'df_analysis' in st.session_state:
    st.write("Preview Data yang sedang aktif:")
    st.dataframe(st.session_state['df_analysis'].head())

    st.markdown("---")

    # --- BAGIAN INPUT ---
    st.header("Pengaturan Analisis")
    st.write("Pilihlah algoritma dan fitur sesuai keingingan anda")
    col1, col2, col3 = st.columns(3)
    with col1:
        algo_choice = st.radio("Pilih Algoritma:", ('K-Means', 'K-Means++', 'OPTICS'))
    with col2:
        all_features = ['ROKOK DAN TEMBAKAU', 'Rokok kretek filter', 'Rokok kretek tanpa filter', 'Rokok putih', 'Tembakau', 'Rokok dan tembakau Lainnya']
        default_features = ['Rokok kretek filter', 'Rokok kretek tanpa filter', 'Rokok putih', 'Tembakau', 'Rokok dan tembakau Lainnya']
        selected_features = st.multiselect("Pilih Fitur Analisis:", options=all_features, default=default_features)
    with col3:
        start_year, end_year = st.slider("Pilih Rentang Tahun:", 2018, 2024, (2018, 2024))
        selected_years = list(range(start_year, end_year + 1))

    st.markdown("---")

    # --- BAGIAN PENGATURAN PARAMETER & PROSES ---
    st.header("Pengaturan Parameter & Klasterisasi")

    params = {}
    if algo_choice in ['K-Means', 'K-Means++']:
        if st.button("Cari rekomendasi nilai K"):
                df_rokok_eval = load_data('dataset_rokok.csv')
                if df_rokok_eval is not None and selected_features:
                    with st.spinner('Mengevaluasi K dari 2-10...'):
                        df_filtered_eval = df_rokok_eval[df_rokok_eval['Tahun'].isin(selected_years)].copy()
                        df_filtered_eval[selected_features] = df_filtered_eval[selected_features].fillna(0)
                        scaler = RobustScaler()
                        X_scaled_eval = scaler.fit_transform(df_filtered_eval[selected_features])

                        k_range = range(2, 11)
                        scores_sklearn = []
                        scores_manual = []
                        for k in k_range:
                            km_sklearn = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
                            scores_sklearn.append(silhouette_score(X_scaled_eval, km_sklearn.fit_predict(X_scaled_eval)))
                            if KMeansPlusPlusManual:
                                km_manual = KMeansPlusPlusManual(n_clusters=k, random_state=42)
                                scores_manual.append(silhouette_score(X_scaled_eval, km_manual.fit_predict(X_scaled_eval)))
                        # Simpan hasil ke session state
                        st.session_state['eval_graph_data'] = {
                        'k_range': k_range,
                        'scores_sklearn': scores_sklearn,
                        'scores_manual': scores_manual
                        }

        help_text = f"Rekomendasi nilai K optimal dapat dilihat pada tombol `Cari rekomendasi nilai K` \n"\
                    f"\n Nilai tertinggi akan memberikan hasil yang paling optimal"\
    
    if 'eval_graph_data' in st.session_state and algo_choice in ['K-Means', 'K-Means++']:
        eval_data = st.session_state['eval_graph_data']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eval_data['k_range'], eval_data['scores_sklearn'], marker='o', linestyle='--', label='K-Means')
        if eval_data['scores_manual']:
            ax.plot(eval_data['k_range'], eval_data['scores_manual'], marker='x', linestyle=':', label='K-Means++')
        ax.set_xlabel("Jumlah Klaster (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Perbandingan Silhouette Score")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        st.info("Nilai K terbaik adalah yang memiliki Silhouette Score tertinggi.")            
        params['k'] = st.slider("Pilih Jumlah Klaster (K):", 2, 10, 3, help=help_text)
        
    elif algo_choice == 'OPTICS': # OPTICS
        # 1. Hitung nilai rekomendasi berdasarkan Aturan #1 (2 * D)
        #    Pastikan selected_features tidak kosong untuk menghindari error
        if selected_features:
            default_min_samples = max(2, len(selected_features) * 2) # max(2, ...) agar tidak pernah < 2
        else:
            default_min_samples = 5 # Nilai aman jika belum ada fitur dipilih

        # 2. Buat Teks Penjelasan (Help Text) untuk User
        help_text = f"Jumlah minimum titik data untuk dianggap sebagai klaster inti. "\
                    f"Nilai rekomendasi (default: {default_min_samples}) dihitung berdasarkan "\
                    f"aturan '2 x Jumlah Fitur' (Anda memilih {len(selected_features)} fitur)." \
                    f"\n\n* Nilai lebih kecil: Lebih sensitif, mendeteksi klaster kecil."\
                    f"\n* Nilai lebih besar: Kurang sensitif, fokus pada klaster besar/padat."
        
        # 3. Gunakan nilai default dan help text di slider
        params['min_samples'] = st.slider(
            "Pilih Jumlah Sampel Minimum (min_samples):", 
            min_value=2, 
            max_value=50, # Anda bisa turunkan maks jadi 30 agar lebih relevan
            value=default_min_samples, 
            help=help_text
        )

        st.caption(f"Rekomendasi `min_samples` untuk {len(selected_features)} fitur adalah **{default_min_samples}**.")
    

if st.button("🚀 Proses Klasterisasi", type="primary", key='process_button'):
        # --- PERBAIKAN: Logika Pengecekan Sebelum Proses ---
        if algo_choice in ['K-Means', 'K-Means++'] and 'eval_graph_data' not in st.session_state:
            st.error("Harap klik tombol 'Cari Rekomendasi Nilai K' terlebih dahulu untuk menampilkan grafik evaluasi sebelum melanjutkan proses klasterisasi.")
        else:
            if 'eval_graph_data' in st.session_state:
                del st.session_state['eval_graph_data']
                
            df_to_process = st.session_state['df_analysis']
            if df_to_process is not None and selected_features:
                with st.spinner('Melakukan analisis...'):
                    df_filtered = df_to_process[df_to_process['Tahun'].isin(selected_years)].copy()
                    start_time = time.time()
                    clustering_results = run_clustering(df_filtered, selected_features, algo_choice, params)
                    runtime = time.time() - start_time
                    
                    if clustering_results:
                        df_result = clustering_results['df_result']
                        df_tambahan = load_data('dataset_tambahan.csv')
                        if df_tambahan is not None:
                            df_tambahan['Tahun'] = df_tambahan['Tahun'].astype(int)
                            df_result['Tahun'] = df_result['Tahun'].astype(int)
                            df_final = pd.merge(df_result, df_tambahan, on=['Kabupaten/Kota', 'Tahun'], how='left')
                        else:
                            df_final = df_result
                        
                        st.session_state['df_final'] = df_final
                        clustering_results.pop('df_result', None)
                        clustering_results['runtime'] = runtime
                        st.session_state['results_info'] = clustering_results
                        st.success("Analisis selesai!")
            else:
                st.warning("Pastikan data telah dimuat dan setidaknya satu fitur dipilih.")

# --- BAGIAN HASIL ANALISIS ---
if 'df_final' in st.session_state:
    results = st.session_state.get('results_info', {})
    df_final = st.session_state.get('df_final')

    if df_final is None or not results:
        st.warning("Hasil analisis tidak tersedia. Silakan jalankan proses klasterisasi.")
        st.stop()

    model = results.get('model')
    features_run = results.get('selected_features_run', selected_features)
    X_scaled_result = results.get('X_scaled')
    df_filtered_result = results.get('df_filtered')

    st.header("📈 Hasil Analisis")
    # Tampilkan Metrik
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score", f"{results.get('sil', 0):.4f}" if 'sil' in results else "N/A")
    col2.metric("Davies-Bouldin Index", f"{results.get('dbi', 0):.4f}" if 'dbi' in results else "N/A")
    col3.metric("Waktu Proses", f"{results.get('runtime', 0):.2f} detik")

    if 'n_clusters_found' in results:
        col1, col2 = st.columns(2)
        col1.metric("Jumlah Klaster Ditemukan", results.get('n_clusters_found', 0))
        col2.metric("Jumlah Data Noise", results.get('n_noise', 0))
    
    st.subheader("Informasi Detail Klaster")
    with st.expander("Klik untuk melihat detail statistik dan teknis klaster"):
        st.write("**Statistik Rata-rata per Klaster:**")
        
        agg_dict = {'ROKOK DAN TEMBAKAU': 'mean', 'Rokok kretek filter' : 'mean', 'Rokok kretek tanpa filter' : 'mean', 'Rokok putih' : 'mean', 'Tembakau' : 'mean', 'Rokok dan tembakau Lainnya' : 'mean'}
        rename_dict = {'ROKOK DAN TEMBAKAU': 'Rata-rata Total Pengeluaran Rokok'}
        if 'IPM' in df_final.columns: agg_dict['IPM'] = 'mean'; rename_dict['IPM'] = 'Rata-rata IPM'
        if 'Persentase_Miskin' in df_final.columns: agg_dict['Persentase_Miskin'] = 'mean'; rename_dict['Persentase_Miskin'] = 'Rata-rata Kemiskinan (%)'

        # Filter noise sebelum menghitung rata-rata
        df_profiling = df_final[df_final['Cluster'] != -1]
        if not df_profiling.empty:
            profiling_data = df_profiling.groupby('Cluster').agg(agg_dict).rename(columns=rename_dict)
            st.dataframe(profiling_data.style.format("{:.2f}"))
        else:
            st.write("Tidak ada klaster valid yang terbentuk untuk dianalisis.")

        st.write("**Informasi Teknis Model:**")
        if hasattr(model, 'n_iter_'):
            st.write(f"- Jumlah Iterasi yang Dibutuhkan: **{model.n_iter_}**")

        st.write("- Posisi Centroid Terakhir (dalam data yang sudah diskalakan):")
        if hasattr(model, 'centroids'):
            centroids_final = model.centroids
            st.dataframe(pd.DataFrame(centroids_final, columns=features_run).style.format("{:.4f}"))
        elif hasattr(model, 'cluster_centers_'):
            centroids_final = model.cluster_centers_
            st.dataframe(pd.DataFrame(centroids_final, columns=features_run).style.format("{:.4f}"))
    

    st.subheader(f"Visualisasi Hasil")
    
    # --- REACHABILITY PLOT (KHUSUS OPTICS) ---
    if 'n_clusters_found' in results: # Tanda bahwa OPTICS dijalankan
        st.write("**Grafik Reachability OPTICS**")
        with st.expander("Klik untuk melihat detail"):
            col1, col2 = st.columns([2, 1])
            with col1:
                model = results.get('model')
                if model:
                    reachability = model.reachability_[model.ordering_]
                    labels = model.labels_[model.ordering_]
                    space = np.arange(len(reachability))

                    # Buat palet warna
                    unique_labels = np.unique(model.labels_)
                    n_clusters_plot = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    cluster_colors = sns.color_palette('viridis', n_clusters_plot)
                    colors = [cluster_colors[label] if label != -1 else (0.5, 0.5, 0.5) for label in labels]

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(space, reachability, color=colors)
                    ax.set_ylabel('Reachability Distance')
                    ax.set_xlabel('Sample Order')
                    ax.set_title('OPTICS Reachability Plot')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    st.pyplot(fig)
            with col2:
                st.write("**Penjelasan Grafik:**")
                st.info("""
                Grafik ini adalah output utama dari OPTICS yang menunjukkan "kepadatan" dari data Anda.
                - **Lembah (Valleys):** Area di mana bar-nya rendah dan memiliki warna yang sama menunjukkan sebuah klaster yang padat.
                - **Puncak (Peaks):** Bar yang tinggi bertindak sebagai pemisah antar klaster.
                - **Bar Abu-abu:** Merupakan data *noise* yang tidak termasuk dalam klaster mana pun.
                """)

    #Tabel Data
    st.write(f"Tabel Data Hasil Klasterisasi")
    with st.expander("Klik untuk melihat detail"):
        display_df = df_final.copy()
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            if 'Provinsi' in display_df.columns:
                provinsi_list = [prov for prov in display_df['Provinsi'].unique().tolist() if isinstance(prov, str)]
                all_provinces = ['Semua Provinsi'] + sorted(provinsi_list)
                selected_province = st.selectbox("Filter berdasarkan Provinsi:", options=all_provinces)
                
                if selected_province != 'Semua Provinsi':
                    display_df = display_df[display_df['Provinsi'] == selected_province]
            else:
                st.write("") # Placeholder

        with filter_col2:
            search_term = st.text_input("Cari berdasarkan Nama Kabupaten/Kota:")
            if search_term:
                display_df = display_df[display_df['Kabupaten/Kota'].str.contains(search_term, case=False, na=False)]
        
        st.dataframe(display_df)

    #Distribusi Klaster
    st.write("**Distribusi Anggota per Klaster**")
    with st.expander("Klik untuk melihat detail"):
        col1, col2 = st.columns([2, 1])
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            order = sorted(df_final['Cluster'].unique())
            labels = ["Noise" if i == -1 else f"Klaster {i}" for i in order]
            sns.countplot(x='Cluster', data=df_final, ax=ax1, palette='viridis', order=order)
            ax1.set_xticklabels(labels)
            st.pyplot(fig1)
        with col2:
            st.write("**Penjelasan Grafik:**")
            st.info("Grafik ini menunjukkan jumlah total data point (kabupaten/kota per tahun) untuk setiap klaster yang terbentuk.")
            cluster_counts = df_final['Cluster'].value_counts().sort_index()
            for i, count in cluster_counts.items():
                label = "Noise" if i == -1 else f"Klaster {i}"
                st.write(f"- {label}: **{count}** anggota")

    #Scatter Plot
    st.write("**Visualisasi Persebaran Klaster**")
    with st.expander("Klik untuk melihat detail"):
        col1, col2 = st.columns([2, 1])
        with col1:
            X_scaled = results.get('X_scaled')
            if X_scaled is not None and len(selected_features) > 1:
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(X_scaled)
                pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                pca_df['Cluster'] = df_final['Cluster'].values
                unique_clusters = sorted(pca_df['Cluster'].unique())
                colors = sns.color_palette('viridis', n_colors=len(unique_clusters) - (1 if -1 in unique_clusters else 0))
                palette = {cluster: (colors[i] if cluster != -1 else (0.5, 0.5, 0.5)) for i, cluster in enumerate(unique_clusters if -1 not in unique_clusters else [c for c in unique_clusters if c != -1])}
                if -1 in unique_clusters: palette[-1] = (0.5, 0.5, 0.5)

                fig3, ax3 = plt.subplots(figsize=(7, 5))
                sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette=palette, data=pca_df, ax=ax3, alpha=0.7, legend='full')
                st.pyplot(fig3)
            else:
                st.warning("PCA tidak dapat ditampilkan (memerlukan minimal 2 fitur).")
        with col2:
            st.write("**Penjelasan Grafik:**")
            st.info("""
            Grafik ini adalah representasi 2D dari data multidimensi Anda. Tujuannya adalah untuk melihat seberapa baik klaster-klaster terpisah secara visual.
            - **Setiap titik:** Mewakili satu data point.
            - **Warna:** Menunjukkan keanggotaan klaster.
            - **Klaster Abu-abu:** Merupakan data *noise* (jika menggunakan OPTICS).
            """)

    #Profil Klaster | Box Plot
    st.write("**Profil Karakteristik Klaster**")
    with st.expander("Klik untuk melihat detail"):
        col1, col2 = st.columns([2, 1])
        with col1:
            df_no_noise = df_final[df_final['Cluster'] != -1]
            if not df_no_noise.empty:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                df_melted = df_no_noise.melt(id_vars=['Cluster'], value_vars=selected_features, var_name='Jenis Rokok', value_name='Nilai Pengeluaran')
                sns.boxplot(x='Jenis Rokok', y='Nilai Pengeluaran', hue='Cluster', data=df_melted, ax=ax2, palette='viridis')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2)
            else:
                st.info("Tidak ada klaster yang terbentuk (semua data dianggap noise).")
        with col2:
            st.write("**Cara Membaca Boxplot:**")
            st.info("""
            - **Garis Tengah:** Median (nilai tengah).
            - **Badan Kotak:** Rentang Interkuartil (IQR), di mana 50% data utama berada.
            - **Garis "Kumis":** Rentang data di luar IQR yang masih dianggap wajar.
            - **Titik-titik:** Outlier (pencilan), yaitu data yang nilainya sangat ekstrem.
            """)

        df_no_noise = df_final[df_final['Cluster'] != -1]
        if not df_no_noise.empty:
            col1, col2 = st.columns(2)
            with col1:
                if 'IPM' in df_final.columns:
                    fig_ipm, ax_ipm = plt.subplots(figsize=(6, 4))
                    sns.boxplot(x='Cluster', y='IPM', data=df_no_noise, ax=ax_ipm, palette='viridis')
                    ax_ipm.set_title("Boxplot IPM")
                    st.pyplot(fig_ipm)
                st.info("Grafik ini membandingkan sebaran Indeks Pembangunan Manusia (IPM) untuk setiap klaster.")
            with col2:
                if 'Persentase_Miskin' in df_final.columns:
                    fig_pov, ax_pov = plt.subplots(figsize=(6, 4))
                    sns.boxplot(x='Cluster', y='Persentase_Miskin', data=df_no_noise, ax=ax_pov, palette='viridis')
                    ax_pov.set_title("Boxplot Tingkat Kemiskinan")
                    st.pyplot(fig_pov)
                st.info("Grafik ini membandingkan sebaran persentase penduduk miskin untuk setiap klaster.")
        else:
            st.info("Tidak ada klaster yang terbentuk untuk dianalisis.")

    #Peta
    st.write("**Pemetaan Geografis Hasil Klasterisasi**")
    with st.expander("Klik untuk melihat detail"):
        # --- PERBAIKAN ERROR StreamlitAPIException ---
        min_map_year = int(df_final['Tahun'].min())
        max_map_year = int(df_final['Tahun'].max())
        
        map_year = 0
        if min_map_year >= max_map_year:
            map_year = min_map_year
            st.info(f"Hanya menampilkan peta untuk satu tahun yang tersedia: {map_year}")
        else:
            map_year = st.slider("Pilih Tahun untuk Peta:", min_map_year, max_map_year, max_map_year)

        with st.spinner("Memuat data peta dan membuat visualisasi..."):
            geojson_path = 'Peta Indonesia Kota Kabupaten simplified.json'
            peta_indo = load_geodata(geojson_path)
            if peta_indo is not None:
                df_peta = df_final[df_final['Tahun'] == map_year]
                if 'NAME_2' in peta_indo.columns:
                    peta_indo['merge_key'] = peta_indo['NAME_2'].str.upper()
                    df_peta['merge_key'] = df_peta['Kabupaten/Kota'].str.upper()
                    peta_merged = peta_indo.merge(df_peta, on='merge_key', how='left')
                    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                    peta_merged.plot(column='Cluster', categorical=True, legend=True,
                                    cmap='viridis', linewidth=0.5, edgecolor='0.8',
                                    missing_kwds={"color": "lightgrey", "label": "Tidak Ada Data"},
                                    ax=ax)
                    ax.set_title(f'Peta Klasterisasi Pengeluaran Rokok per Kab/Kota - Tahun {map_year}', fontsize=16)
                    ax.set_axis_off()
                    st.pyplot(fig)
                else:
                    st.error("Kolom 'NAME_2' tidak ditemukan di file GeoJSON.")
            else:
                st.error("Gagal membuat peta karena file GeoJSON tidak dapat dimuat.")

    #Perkembangan Pengeluaran Rokok
    st.write("**Perkembangan Rata-rata Total Pengeluaran Rokok per Klaster**")
    with st.expander("Klik untuk melihat detail"):
        col1, col2 = st.columns([2, 1])
        with col1:
            if 'ROKOK DAN TEMBAKAU' in df_final.columns:
                df_no_noise = df_final[df_final['Cluster'] != -1]
                if not df_no_noise.empty:
                    avg_expenditure = df_no_noise.groupby(['Cluster', 'Tahun'])['ROKOK DAN TEMBAKAU'].mean().reset_index()
                    
                    fig_line, ax_line = plt.subplots(figsize=(10, 5))
                    sns.lineplot(data=avg_expenditure, x='Tahun', y='ROKOK DAN TEMBAKAU', hue='Cluster', palette='viridis', marker='o', ax=ax_line)
                    ax_line.set_title('Rata-rata Total Pengeluaran Rokok per Klaster per Tahun')
                    ax_line.set_ylabel('Rata-rata Pengeluaran')
                    ax_line.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig_line)
                else:
                    st.info("Tidak ada klaster valid untuk menampilkan tren tahunan.")
            else:
                st.warning("Kolom 'ROKOK DAN TEMBAKAU' tidak ditemukan dalam data.")
        with col2:
            st.write("**Penjelasan Grafik:**")
            st.info("""
            Grafik garis ini menunjukkan bagaimana rata-rata total pengeluaran rokok dan tembakau berubah dari tahun ke tahun untuk setiap klaster.
            - **Sumbu X:** Tahun.
            - **Sumbu Y:** Rata-rata pengeluaran.
            - **Garis Berwarna:** Mewakili setiap klaster yang ditemukan.
            Ini membantu Anda melihat tren naik/turun pengeluaran untuk setiap segmen pasar.
        """)

    #Top 10 Barchart
    st.write("**Top 10 Daerah dengan Pengeluaran Tertinggi**")
    with st.expander("Klik untuk melihat detail"):
        df_no_noise_for_palette = df_final[df_final['Cluster'] != -1]
        valid_clusters_master = sorted(df_no_noise_for_palette['Cluster'].unique())
        master_color_palette = sns.color_palette('viridis', n_colors=len(valid_clusters_master))
        cluster_to_color_map = {cluster: master_color_palette[i] for i, cluster in enumerate(valid_clusters_master)}

        if not df_final.empty:
            for year in sorted(df_final['Tahun'].unique()):
                with st.expander(f"Tahun {year}"):
                    df_year = df_final[df_final['Tahun'] == year]
                    for feature in selected_features:
                        st.write(f"**{feature}**")
                        top_10_df = df_year.nlargest(10, feature)
                        if not top_10_df.empty:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plot_df = top_10_df.set_index('Kabupaten/Kota')
                            
                            # Bangun palet spesifik untuk plot ini
                            hue_order_plot = sorted(plot_df['Cluster'].unique())
                            palette_plot = {}
                            for cluster in hue_order_plot:
                                if cluster == -1:
                                    palette_plot[cluster] = (0.5, 0.5, 0.5) # Abu-abu untuk noise
                                else:
                                    palette_plot[cluster] = cluster_to_color_map.get(cluster, (0,0,0)) # Ambil dari master map

                            sns.barplot(data=plot_df, x=plot_df.index, y=feature, hue='Cluster', palette=palette_plot, ax=ax, dodge=False)
                            ax.set_title(f'Top 10 Pengeluaran Tertinggi - {feature} ({year})')
                            ax.set_xlabel("Kabupaten/Kota")
                            ax.set_ylabel("Nilai Pengeluaran")
                            plt.xticks(rotation=45, ha='right')
                            ax.legend(title='Klaster')
                            st.pyplot(fig)


        
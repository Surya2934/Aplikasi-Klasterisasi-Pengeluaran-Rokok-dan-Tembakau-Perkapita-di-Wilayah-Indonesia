import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import numpy as np
import geopandas as gpd 
import plotly.express as px
import folium
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from streamlit_folium import st_folium
from sklearn.metrics import silhouette_samples

# --- Import kode K-Means++ Manual ---
try:
    from k_means_plus_plus import KMeansPlusPlusManual
except ImportError:
    st.error("File 'k_means_plus_plus.py' tidak ditemukan.")
    KMeansPlusPlusManual = None

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Utama", layout="wide")

with st.sidebar:
    st.title("Dashboard Klasterisasi")

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

# --- BAGIAN INPUT DATA ---
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
            # Muat data asli (dataset_rokok.csv)
            df_rokok_eval = load_data('dataset_rokok.csv')
            if df_rokok_eval is not None and selected_features:
                with st.spinner('Mengevaluasi K dari 2-10 pada data...'):
                    # 1. Filter data asli berdasarkan rentang tahun
                    df_filtered_eval_raw = df_rokok_eval[df_rokok_eval['Tahun'].isin(selected_years)].copy()

                    #Mengagregasi data evaluasi
                    features_to_agg_eval = selected_features
                    df_aggregated_eval = df_filtered_eval_raw.groupby('Kabupaten/Kota')[features_to_agg_eval].median().reset_index()

                    #Memastikan tidak ada NaN setelah agregasi
                    df_aggregated_eval[features_to_agg_eval] = df_aggregated_eval[features_to_agg_eval].fillna(0)

                    # 2. Scaling pada data agregat
                    scaler_eval = RobustScaler()
                    # Scale HANYA fitur yang dipilih dari data agregat
                    X_scaled_eval = scaler_eval.fit_transform(df_aggregated_eval[features_to_agg_eval])

                    # 3. Hitung Silhouette Score pada data agregat
                    k_range = range(2, 11)
                    scores_sklearn = []
                    scores_manual = []
                    for k in k_range:
                        # K-Means (Sklearn)
                        km_sklearn = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
                        # Gunakan X_scaled_eval (data agregat)
                        labels_sklearn = km_sklearn.fit_predict(X_scaled_eval)
                        # Hindari error jika hanya 1 klaster terbentuk (jarang terjadi tapi mungkin)
                        if len(set(labels_sklearn)) > 1:
                             scores_sklearn.append(silhouette_score(X_scaled_eval, labels_sklearn))
                        else:
                             scores_sklearn.append(-1) # Skor tidak valid

                        # K-Means++ (Manual)
                        if KMeansPlusPlusManual:
                            km_manual = KMeansPlusPlusManual(n_clusters=k, random_state=42)
                            # Gunakan X_scaled_eval (data agregat)
                            labels_manual = km_manual.fit_predict(X_scaled_eval)
                            if len(set(labels_manual)) > 1:
                                scores_manual.append(silhouette_score(X_scaled_eval, labels_manual))
                            else:
                                scores_manual.append(-1) # Skor tidak valid

                    # 4. Simpan hasil ke session state
                    st.session_state['eval_graph_data'] = {
                        'k_range': list(k_range),
                        'scores_sklearn': scores_sklearn,
                        'scores_manual': scores_manual
                    }
                    st.success("Evaluasi K selesai.")
            elif df_rokok_eval is None:
                 st.error("Gagal memuat dataset_rokok.csv untuk evaluasi.")
            else:
                 st.warning("Pilih setidaknya satu fitur untuk evaluasi K.")

        help_text = f"Rekomendasi nilai K optimal dapat dilihat pada tombol `Cari rekomendasi nilai K` \n"\
                    f"\n Nilai tertinggi akan memberikan hasil yang paling optimal"\
    
        if 'eval_graph_data' in st.session_state:
            eval_data = st.session_state['eval_graph_data']

            #Membuat DataFrame agar lebih mudah diplot dengan Plotly
            k_range = eval_data['k_range']
            scores_sklearn = eval_data['scores_sklearn']
            scores_manual = eval_data['scores_manual']

            #Buat list of dictionaries
            plot_data = []
            for i, k in enumerate(k_range):
                if i < len(scores_sklearn): #Memastikan index valid
                    plot_data.append({'Jumlah Klaster (K)': k, 'Silhouette Score': scores_sklearn[i], 'Algoritma': 'K-Means'})
                if scores_manual and i < len(scores_manual): #Cek jika K-Means++ dievaluasi dan index valid
                    plot_data.append({'Jumlah Klaster (K)': k, 'Silhouette Score': scores_manual[i], 'Algoritma': 'K-Means++'})

            #Konversi ke DataFrame
            eval_df = pd.DataFrame(plot_data)

            #Membuat plot dengan plotly
            if not eval_df.empty:
                fig_eval = px.line(
                    eval_df,
                    x='Jumlah Klaster (K)',
                    y='Silhouette Score',
                    color='Algoritma',
                    markers=True,
                    title="Perbandingan Silhouette Score"
                )
                fig_eval.update_layout(xaxis_title="Jumlah Klaster (K)", yaxis_title="Silhouette Score")
                st.plotly_chart(fig_eval, use_container_width=True) # <-- Tampilkan plot Plotly
                st.info("Nilai K terbaik adalah yang memiliki Silhouette Score tertinggi. Hover di atas titik untuk melihat nilai pasti.")
            else:
                st.warning("Data evaluasi tidak tersedia untuk ditampilkan.")
            
        help_text_kmeans = "Anda dapat melihat rekomendasi nilai K dengan menekan tombol di atas."
        k_value_default = st.session_state.get('results_info', {}).get('k', 3)
        params['k'] = st.slider("Pilih Jumlah Klaster (K):", 2, 10, k_value_default, help=help_text_kmeans)
        
    elif algo_choice == 'OPTICS': # OPTICS
        #1. Hitung nilai rekomendasi berdasarkan Aturan #1 (2 * D)
        #Memastikan selected_features tidak kosong untuk menghindari error
        if selected_features:
            default_min_samples = max(2, len(selected_features) * 2)
        else:
            default_min_samples = 5

        #2. Buat Teks Penjelasan (Help Text)
        help_text = f"Jumlah minimum titik data untuk dianggap sebagai klaster inti. "\
                    f"Nilai rekomendasi (default: {default_min_samples}) dihitung berdasarkan "\
                    f"aturan '2 x Jumlah Fitur' (Anda memilih {len(selected_features)} fitur)." \
                    f"\n\n* Nilai lebih kecil: Lebih sensitif, mendeteksi klaster kecil."\
                    f"\n* Nilai lebih besar: Kurang sensitif, fokus pada klaster besar/padat."
        
        #3. Gunakan nilai default dan help text di slider
        params['min_samples'] = st.slider(
            "Pilih Jumlah Sampel Minimum (min_samples):", 
            min_value=2, 
            max_value=50,
            value=default_min_samples, 
            help=help_text
        )

        st.caption(f"Rekomendasi `min_samples` untuk {len(selected_features)} fitur adalah **{default_min_samples}**.")
    
if st.button("🚀 Proses Klasterisasi", type="primary", key='process_button'):
        df_to_process = st.session_state['df_analysis']
        if df_to_process is not None and selected_features:
            with st.spinner('Melakukan analisis...'):
                
                # 1. Filter data asli berdasarkan rentang tahun
                df_filtered = df_to_process[df_to_process['Tahun'].isin(selected_years)].copy()
                
                # 2. LAKUKAN AGREGRASI MEDIAN PER KABUPATEN/KOTA
                features_to_agg = selected_features
                df_aggregated = df_filtered.groupby('Kabupaten/Kota')[features_to_agg].median().reset_index()
                
                # 3. JALANKAN KLASTERISASI PADA DATA AGREGAT
                start_time = time.time()
                clustering_results = run_clustering(df_aggregated, selected_features, algo_choice, params)
                runtime = time.time() - start_time
                
                if clustering_results:
                    # 4. Dapatkan hasil dari data agregat
                    df_result_agg = clustering_results['df_result']
                    
                    # 5. Buat peta (dictionary) dari hasil klasterisasi
                    cluster_map = df_result_agg.set_index('Kabupaten/Kota')['Cluster'].to_dict()

                    # 6. Terapkan (map) label klaster kembali ke data asli (df_filtered)
                    df_final_with_clusters = df_filtered.copy()
                    df_final_with_clusters['Cluster'] = df_final_with_clusters['Kabupaten/Kota'].map(cluster_map)
                    
                    # 7. Gabungkan dengan data tambahan (IPM, Kemiskinan, dll)
                    df_tambahan = load_data('dataset_tambahan.csv')
                    if df_tambahan is not None:
                        df_tambahan['Tahun'] = df_tambahan['Tahun'].astype(int)
                        df_final_with_clusters['Tahun'] = df_final_with_clusters['Tahun'].astype(int)
                        df_final = pd.merge(df_final_with_clusters, df_tambahan, on=['Kabupaten/Kota', 'Tahun'], how='left')
                    else:
                        df_final = df_final_with_clusters
                    
                    # 8. Simpan data final (yang besar) ke session state
                    st.session_state['df_final'] = df_final
                    
                    clustering_results['runtime'] = runtime
                    clustering_results['selected_features_run'] = selected_features
                    clustering_results['algo_run'] = algo_choice 
                    clustering_results['start_year'] = start_year 
                    clustering_results['end_year'] = end_year    
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

# --- Definisikan Skema Warna & Label Terpusat (Persiapan Plotly) ---

    # 1. Tentukan palet warna
    df_non_noise = df_final[df_final['Cluster'] != -1]
    unique_clusters = sorted(df_non_noise['Cluster'].unique())
    
    colors_sns = sns.color_palette('viridis', n_colors=len(unique_clusters))
    colors_hex = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for (r,g,b) in colors_sns]

    # 3. Buat Peta Label (master_label_map)
    master_label_map = {cluster: f"Klaster {cluster}" for cluster in unique_clusters}
    master_label_map[-1] = "Noise"

    # 2. Buat Peta Warna (master_color_map)
    # Kita ambil label dari master_label_map yang baru dibuat
    master_color_map = {master_label_map[cluster]: colors_hex[i] for i, cluster in enumerate(unique_clusters)}
    master_color_map["Noise"] = 'grey'
    master_color_map["Tidak Ada Data"] = '#D3D3D3'

    # 4. Terapkan Label ke DataFrame
    df_final['Cluster_Label'] = df_final['Cluster'].map(master_label_map)
    
    # Menerapkan ke data agregat
    df_result_agg = results.get('df_result')
    if df_result_agg is not None:
        df_result_agg['Cluster_Label'] = df_result_agg['Cluster'].map(master_label_map)
        # Simpan kembali ke results_info untuk Scatter Plot
        st.session_state['results_info']['df_result'] = df_result_agg

    # Tampilkan Informasi
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
        if 'Persentase_Miskin' in df_final.columns: agg_dict['Persentase_Miskin'] = 'mean'; rename_dict['Miskin (%)'] = 'Rata-rata Kemiskinan (%)'

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
    
    #Tabel Data
    st.write(f"Tabel Data Hasil Klasterisasi")
    with st.expander("Klik untuk melihat detail"):
        display_df = df_final.copy()
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            if 'Provinsi' in display_df.columns:
                provinsi_list = [prov for prov in display_df['Provinsi'].unique().tolist() if isinstance(prov, str)]
                all_provinces = ['Semua Provinsi'] + sorted(provinsi_list)
                selected_province = st.selectbox("Filter berdasarkan Provinsi:", options=all_provinces)
                
                if selected_province != 'Semua Provinsi':
                    display_df = display_df[display_df['Provinsi'] == selected_province]
            else:
                st.write("")

        with filter_col2:
            search_term = st.text_input("Cari berdasarkan Nama Kabupaten/Kota:")
            if search_term:
                display_df = display_df[display_df['Kabupaten/Kota'].str.contains(search_term, case=False, na=False)]
        
        with filter_col3:
            cluster_labels_unique = sorted(display_df['Cluster'].unique())
            cluster_options_map = {}
            for c in cluster_labels_unique:
                if c == -1: cluster_options_map["Noise"] = -1
                elif c == -2: cluster_options_map["Gagal Merge"] = -2
                else: cluster_options_map[f"Klaster {c}"] = c
            
            all_cluster_options = ['Semua Klaster'] + list(cluster_options_map.keys())
            selected_cluster_label = st.selectbox("Filter Klaster:", options=all_cluster_options, key="filter_cluster")

            if selected_cluster_label != 'Semua Klaster':
                selected_cluster_num = cluster_options_map[selected_cluster_label]
                display_df = display_df[display_df['Cluster'] == selected_cluster_num]

        columns_to_hide = ['Cluster_Label', 'key_df']
        
        cols_exist_to_hide = [col for col in columns_to_hide if col in display_df.columns]
        
        # Drop kolom yang ada
        if cols_exist_to_hide:
            df_to_display = display_df.drop(columns=cols_exist_to_hide)
        else:
            df_to_display = display_df

        df_to_display = df_to_display.reset_index(drop=True)

        st.dataframe(df_to_display)

    st.subheader(f"Visualisasi Hasil")
    
    #Siluet plot
    st.write("**Visualisasi Silhouette per Klaster**")
    with st.expander("Klik untuk melihat detail"):
        col1_sil, col2_sil = st.columns([2,1]) # Buat kolom untuk plot dan penjelasan
        with col1_sil:
            # Ambil data agregat dan hasil klasterisasinya
            X_scaled_agg = results.get('X_scaled')
            df_result_agg = results.get('df_result') # Ini adalah df agregat dgn cluster

            if X_scaled_agg is not None and df_result_agg is not None:
                labels = df_result_agg['Cluster'].values
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # Hitung jumlah klaster (abaikan noise -1)
                avg_silhouette_score = results.get('sil') # Ambil skor rata-rata yg sudah dihitung

                if n_clusters > 1 and avg_silhouette_score is not None: # Perlu > 1 klaster
                    fig_sil, ax_sil = plt.subplots(1, 1, figsize=(7, 5))

                    # Batas bawah y-axis untuk plot klaster pertama
                    y_lower = 10

                    # Hitung skor siluet untuk setiap sampel
                    sample_silhouette_values = silhouette_samples(X_scaled_agg, labels)

                    for i in range(n_clusters):
                        # Ambil skor siluet untuk sampel di klaster i, dan urutkan
                        ith_cluster_silhouette_values = \
                            sample_silhouette_values[labels == i]
                        ith_cluster_silhouette_values.sort()

                        size_cluster_i = ith_cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i

                        # Pilih warna untuk klaster
                        color = cm.viridis(float(i) / n_clusters) # Gunakan colormap viridis
                        ax_sil.fill_betweenx(np.arange(y_lower, y_upper),
                                          0, ith_cluster_silhouette_values,
                                          facecolor=color, edgecolor=color, alpha=0.7)

                        # Label di tengah plot siluet klaster
                        ax_sil.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                        # Hitung batas y bawah baru untuk plot berikutnya
                        y_lower = y_upper + 10  # 10 untuk spasi antar plot

                    ax_sil.set_title("Silhouette Plot untuk Setiap Klaster")
                    ax_sil.set_xlabel("Nilai Koefisien Silhouette")
                    ax_sil.set_ylabel("Label Klaster")

                    # Garis vertikal untuk rata-rata skor siluet
                    ax_sil.axvline(x=avg_silhouette_score, color="red", linestyle="--", label=f"Avg: {avg_silhouette_score:.2f}")
                    ax_sil.legend(loc='best')

                    ax_sil.set_yticks([])
                    ax_sil.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                    st.pyplot(fig_sil)

                elif avg_silhouette_score is None:
                     st.warning("Skor Silhouette rata-rata tidak tersedia.")
                else:
                     st.info("Silhouette plot memerlukan setidaknya 2 klaster untuk ditampilkan.")
            else:
                 st.warning("Data agregat hasil klasterisasi tidak ditemukan.")

        with col2_sil:
            st.write("**Cara Membaca Silhouette Plot:**")
            st.info("""
            Plot ini membantu mengevaluasi seberapa baik setiap titik data cocok dalam klasternya dibandingkan dengan klaster tetangga.

            * **Lebar Bentuk (Sumbu X):** Menunjukkan nilai koefisien *silhouette*. Semakin mendekati +1, semakin baik titik data tersebut cocok di klasternya dan jauh dari klaster lain. Nilai mendekati 0 berarti dekat dengan batas antar klaster. Nilai negatif berarti mungkin salah klaster.
            * **Tinggi Bentuk (Sumbu Y):** Menunjukkan jumlah titik data dalam klaster tersebut.
            * **Garis Merah Putus-putus:** Menunjukkan rata-rata koefisien *silhouette* untuk *semua* titik data.
            * **Bentuk Ideal:** Bentuk yang "gemuk" (lebar merata mendekati 1) dan tingginya proporsional menunjukkan klaster yang bagus. Bentuk yang "tipis" atau puncaknya jauh di bawah garis merah menandakan klaster yang kurang baik.
            """)

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

                    # Buat DataFrame untuk Plotly
                    df_reach = pd.DataFrame({
                        'Sample Order': space,
                        'Reachability Distance': reachability,
                        'Cluster': labels
                    })
                    # Map ke label yang sudah dibuat
                    df_reach['Cluster_Label'] = df_reach['Cluster'].map(master_label_map)

                    fig_reach = px.bar(
                        df_reach,
                        x='Sample Order',
                        y='Reachability Distance',
                        color='Cluster_Label',
                        color_discrete_map=master_color_map,
                        title='OPTICS Reachability Plot'
                    )
                    
                    # Sembunyikan legend (terlalu ramai)
                    fig_reach.update_layout(showlegend=False, dragmode='pan') 
                    
                    st.plotly_chart(fig_reach, use_container_width=True)
            with col2:
                st.write("**Penjelasan Grafik:**")
                st.info("""
                Grafik ini menunjukkan "kepadatan" data Anda. **Hover** untuk melihat nilai reachability.
                - **Lembah (Valleys):** Area rendah menunjukkan klaster padat.
                - **Puncak (Peaks):** Batang tinggi pemisah antar klaster.
                - **Bar Abu-abu:** Data *noise*.
                """)

    #Distribusi Klaster
    st.write("**Distribusi Anggota per Klaster**")
    with st.expander("Klik untuk melihat detail"):
        col1, col2 = st.columns([2, 1])
        
        df_result_agg = results.get('df_result') 
        
        with col1:
            if df_result_agg is not None:
                # Gunakan df_result_agg yang sudah memiliki 'Cluster_Label'
                counts_df = df_result_agg.groupby('Cluster_Label').size().reset_index(name='Jumlah')
                
                fig1 = px.bar(counts_df, 
                             x='Cluster_Label', 
                             y='Jumlah', 
                             color='Cluster_Label',
                             color_discrete_map=master_color_map,
                             title="Distribusi Anggota per Klaster"
                            )
                fig1.update_layout(xaxis_title="Klaster", yaxis_title="Jumlah Kabupaten/Kota", dragmode='pan')
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("Data tidak ditemukan.")
                
        with col2:
            st.write("**Penjelasan Grafik:**")
            st.info("""
            Grafik ini menunjukkan jumlah **Kabupaten/Kota** untuk setiap klaster. 
            **Hover** di atas bar untuk melihat jumlah pastinya.
            """)
            
            if df_result_agg is not None:
                cluster_counts = df_result_agg['Cluster_Label'].value_counts().sort_index()
                for label, count in cluster_counts.items():
                    st.write(f"- {label}: **{count}** anggota (Kab/Kota)")
            else:
                st.write("Tidak ada data untuk dihitung.")

    #Scatter Plot
    st.write("**Visualisasi Persebaran Klaster**")
    with st.expander("Klik untuk melihat detail"):
        col1, col2 = st.columns([2, 1])
        with col1:
            X_scaled_agg = results.get('X_scaled') 
            df_result_agg = results.get('df_result') # Ambil df_result_agg yg sdh update
            
            if X_scaled_agg is not None and df_result_agg is not None and len(features_run) > 1:
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(X_scaled_agg)
                pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                
                # Tambahkan label dan nama kab/kota untuk hover
                pca_df['Cluster_Label'] = df_result_agg['Cluster_Label'].values 
                pca_df['Kabupaten/Kota'] = df_result_agg['Kabupaten/Kota'].values

                fig3 = px.scatter(pca_df, 
                                  x='PC1', 
                                  y='PC2', 
                                  color='Cluster_Label',
                                  color_discrete_map=master_color_map,
                                  title='Visualisasi Persebaran Klaster',
                                  custom_data=['Kabupaten/Kota', 'Cluster_Label'])
                
                fig3.update_traces(
                    hovertemplate="<b>Kab/Kota:</b> %{customdata[0]}<br>" +
                                  "<b>Klaster:</b> %{customdata[1]}<br>" +
                                  "<extra></extra>" # Hapus info tambahan default
                )

                fig3.update_layout(legend_title_text='Klaster', dragmode='pan')
                st.plotly_chart(fig3, use_container_width=True)
            
            elif X_scaled_agg is None or df_result_agg is None:
                    st.warning("Data hasil tidak ditemukan.")
            else:
                st.warning("Visualisasi tidak dapat ditampilkan (memerlukan minimal 2 fitur).")
        with col2:
            st.write("**Penjelasan Grafik:**")
            st.info("""
            Grafik ini adalah representasi 2D dari data Anda. 
            **Hover** di atas titik untuk melihat nama **Kabupaten/Kota**.
            """)

    #Profil Klaster | Box Plot
    st.write("**Profil Karakteristik Klaster**")
    with st.expander("Klik untuk melihat detail"):
        # --- Boxplot untuk fitur-fitur Rokok ---
        col1, col2 = st.columns([2, 1])
        with col1:
            # Gunakan df_final (DataFrame LENGKAP)
            if not df_final.empty:
                # Memastikan 'Cluster_Label' masuk ke id_vars
                # Melt dari df_final
                df_melted_all = df_final.melt(
                    id_vars=['Cluster_Label'],
                    value_vars=features_run,
                    var_name='Jenis Rokok',
                    value_name='Nilai Pengeluaran'
                 )

                fig2_all = px.box(df_melted_all,
                              x='Jenis Rokok',
                              y='Nilai Pengeluaran',
                              color='Cluster_Label',
                              color_discrete_map=master_color_map,
                              title="Profil Karakteristik Klaster")
                fig2_all.update_layout(legend_title_text='Klaster', dragmode='pan')
                fig2_all.update_xaxes(tickangle=45)
                st.plotly_chart(fig2_all, use_container_width=True)
            else:
                st.info("Tidak ada data untuk menampilkan boxplot.")
        with col2:
            st.write("**Cara Membaca Boxplot:**")
            st.info("""
            Grafik ini menunjukkan sebaran data pengeluaran untuk setiap jenis rokok, dikelompokkan berdasarkan klaster.
            **Hover** di atas kotak untuk detail.
            """)

        if not df_final.empty:
            col1_extra, col2_extra = st.columns(2)
            with col1_extra:
                if 'IPM' in df_final.columns:
                    fig_ipm_all = px.box(df_final, # <-- Gunakan df_final
                                     x='Cluster_Label',
                                     y='IPM',
                                     color='Cluster_Label',
                                     color_discrete_map=master_color_map,
                                     title="Boxplot IPM per Klaster")
                    fig_ipm_all.update_layout(xaxis_title="Klaster/Noise", showlegend=False, dragmode='pan')
                    st.plotly_chart(fig_ipm_all, use_container_width=True)
                    st.info("Grafik perbandingan sebaran IPM.")
                else:
                    st.warning("Kolom 'IPM' tidak ditemukan.")

            with col2_extra:
                if 'Persentase_Miskin' in df_final.columns:
                    fig_pov_all = px.box(df_final, # <-- Gunakan df_final
                                     x='Cluster_Label',
                                     y='Persentase_Miskin',
                                     color='Cluster_Label',
                                     color_discrete_map=master_color_map,
                                     title="Boxplot Kemiskinan per Klaster")
                    fig_pov_all.update_layout(xaxis_title="Klaster/Noise", showlegend=False, dragmode='pan')
                    st.plotly_chart(fig_pov_all, use_container_width=True)
                    st.info("Grafik perbandingan sebaran kemiskinan.")
                else:
                     st.warning("Kolom 'Persentase_Miskin' tidak ditemukan.")
        else:
            st.info("Tidak ada data untuk dianalisis.")

    #Peta
    st.write("**Pemetaan Geografis Hasil Klasterisasi (Folium - Koreksi Warna)**")
    with st.expander("Klik untuk melihat detail"):
        min_map_year = int(df_final['Tahun'].min())
        max_map_year = int(df_final['Tahun'].max())

        map_year = 0
        if min_map_year >= max_map_year:
            map_year = min_map_year
            st.info(f"Hanya menampilkan peta untuk satu tahun yang tersedia: {map_year}")
        else:
            map_year = st.slider("Pilih Tahun untuk Peta:", min_map_year, max_map_year, max_map_year, key="folium_year_slider_colored")

        with st.spinner("Memuat data peta dan membuat visualisasi..."):
            geojson_path = 'Peta Indonesia Kota Kabupaten simplified.json'

            @st.cache_data
            def load_and_prepare_geojson_folium_colored(path):
                gdf = load_geodata(path)
                if gdf is None: return None
                if 'NAME_2' not in gdf.columns:
                    st.error("Kolom 'NAME_2' tidak ditemukan di file GeoJSON.")
                    return None
                try:
                    if gdf.crs is None:
                        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

                    elif gdf.crs.to_epsg() != 4326:
                        gdf = gdf.to_crs("EPSG:4326")
                    
                    gdf['key_geojson'] = gdf['NAME_2'].str.upper()
                    gdf = gdf[['key_geojson', 'NAME_2', 'geometry']]
                    return gdf
                except Exception as e:
                    st.error(f"Error saat memproses CRS/Key GeoJSON: {e}")
                    return None

            peta_indo_gdf = load_and_prepare_geojson_folium_colored(geojson_path)

            if peta_indo_gdf is not None:

                # 1. Siapkan data CSV
                if 'key_df' not in df_final.columns:
                     df_final['key_df'] = df_final['Kabupaten/Kota'].str.upper()
                if 'Cluster_Label' not in df_final.columns:
                    st.error("Kolom 'Cluster_Label' tidak ditemukan di df_final.")
                    st.stop()

                df_peta = df_final[df_final['Tahun'] == map_year].copy()
                
                # Persiapan Tooltip
                tooltip_fields = ['Kabupaten/Kota', 'Cluster_Label', 'Provinsi']
                tooltip_aliases = ['Kabupaten/Kota:', 'Klaster:', 'Provinsi:']
                def safe_to_str(series, decimals=2, default_val="N/A"):
                    try:
                        numeric_series = pd.to_numeric(series, errors='coerce')
                        return numeric_series.round(decimals).astype(str).fillna(default_val)
                    except Exception:
                         return pd.Series([default_val] * len(series), index=series.index)
                if 'IPM' in df_peta.columns:
                    df_peta['IPM_str'] = safe_to_str(df_peta['IPM'])
                    tooltip_fields.append('IPM_str'); tooltip_aliases.append('IPM:')
                if 'Persentase_Miskin' in df_peta.columns:
                    df_peta['Miskin_str'] = safe_to_str(df_peta['Persentase_Miskin'])
                    tooltip_fields.append('Miskin_str'); tooltip_aliases.append('Kemiskinan (%):')
                if 'ROKOK DAN TEMBAKAU' in df_peta.columns:
                     df_peta['Rokok_str'] = safe_to_str(df_peta['ROKOK DAN TEMBAKAU'])
                     tooltip_fields.append('Rokok_str'); tooltip_aliases.append('Total Pengeluaran Rokok:')

                # 2. Buat Peta Dasar Folium
                map_center = [-2.5, 118.0]
                m = folium.Map(location=map_center, zoom_start=5, tiles='cartodbpositron')

                # 3. Gabungkan data ke GeoDataFrame
                merged_gdf = peta_indo_gdf.merge(
                    df_peta,
                    left_on='key_geojson',
                    right_on='key_df',
                    how='left'
                )
                merged_gdf['Cluster_Label'] = merged_gdf['Cluster_Label'].fillna('Tidak Ada Data').astype(str)
                merged_gdf['Kabupaten/Kota'] = merged_gdf['Kabupaten/Kota'].fillna(merged_gdf['NAME_2'])
                for col in tooltip_fields:
                    if col not in merged_gdf.columns: merged_gdf[col] = 'N/A'
                    elif '_str' in col: merged_gdf[col] = merged_gdf[col].fillna('N/A')


                # 4. Buat Style Function
                def style_function(feature):
                    label = 'Tidak Ada Data'
                    properties = feature.get('properties', {}) 
                    if properties: 
                        label = properties.get('Cluster_Label', 'Tidak Ada Data')
                    
                    fill_color = master_color_map.get(label, '#D3D3D3') 

                    return {
                        'fillColor': fill_color,
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.7 if label != 'Tidak Ada Data' else 0.3
                    }

                # 5. Tambahkan Layer GeoJson dengan Style dan Tooltip
                try:
                    folium.GeoJson(
                        merged_gdf,
                        style_function=style_function,
                        tooltip=folium.features.GeoJsonTooltip(
                            fields=tooltip_fields,
                            aliases=tooltip_aliases,
                            localize=True, sticky=False, labels=True,
                            style="background-color: #F0EFEF; border: 2px solid black; border-radius: 3px; box-shadow: 3px;",
                            max_width=800,
                        ),
                        highlight_function=lambda x: {'weight':3, 'fillOpacity':1},
                        name=f'Peta Klaster Tahun {map_year}'
                    ).add_to(m)

                    legend_html = f'''
                         <div style="position: absolute; 
                         bottom: 50px; left: 50px; width: auto; height: auto; 
                         background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
                         border-radius: 6px; padding: 10px;
                         color: black; 
                         ">
                         <b>Legenda Klaster</b><br style="margin-bottom: 5px;">
                      '''
                    
                    # 2. Logika sorting
                    sorted_items = sorted(master_color_map.items(), key=lambda item: (
                        'Z' if item[0] in ['Tidak Ada Data', 'Noise'] else item[0], item[0]
                    ))

                    # 3. Loop untuk menambahkan baris legenda
                    for label, color in sorted_items:
                        opacity = 0.3 if label == 'Tidak Ada Data' else 0.7
                        
                        # Tambahkan baris (flexbox)
                        legend_html += '<div style="display: flex; align-items: center; margin-bottom: 4px;">'
                        
                        # Tambahkan kotak warna
                        legend_html += f'<i style="background: {color}; width: 18px; height: 18px; margin-right: 8px; border: 1px solid black; opacity: {opacity};"></i>'
                        
                        # Tambahkan label teks
                        legend_html += f'<span>{label}</span>'
                        
                        # Tutup baris
                        legend_html += '</div>'
                    
                    # 4. Tutup div utama legenda
                    legend_html += '</div>'
                    
                    # 5. Tambahkan elemen HTML ke peta
                    m.get_root().html.add_child(folium.Element(legend_html))

                    # 6. Tampilkan di Streamlit
                    st_folium(m, width='100%', height=500)

                except Exception as e:
                    st.error(f"Terjadi error saat membuat peta Folium berwarna: {e}")
                    st.write("Tipe data GDF hasil merge:")
                    st.dataframe(merged_gdf.dtypes.apply(lambda x: x.name))

            else:
                st.error("Gagal memuat/memproses file GeoJSON untuk Folium.")
                
    #Perkembangan Pengeluaran Rokok
    st.write("**Perkembangan Rata-rata Total Pengeluaran Rokok per Klaster**")
    with st.expander("Klik untuk melihat detail"):
        col1, col2 = st.columns([2, 1])
        with col1:
            if 'ROKOK DAN TEMBAKAU' in df_final.columns:
                df_no_noise = df_final[df_final['Cluster'] != -1]
                if not df_no_noise.empty:
                    # Group by pakai 'Cluster_Label'
                    avg_expenditure = df_no_noise.groupby(['Cluster_Label', 'Tahun'])['ROKOK DAN TEMBAKAU'].mean().reset_index()
                    
                    fig_line = px.line(avg_expenditure,
                                       x='Tahun',
                                       y='ROKOK DAN TEMBAKAU',
                                       color='Cluster_Label',
                                       color_discrete_map=master_color_map,
                                       markers=True,
                                       title='Rata-rata Total Pengeluaran Rokok per Klaster per Tahun')
                    
                    fig_line.update_layout(yaxis_title='Rata-rata Pengeluaran', legend_title_text='Klaster', dragmode='pan')
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info("Tidak ada klaster valid untuk menampilkan tren tahunan.")
            else:
                st.warning("Kolom 'ROKOK DAN TEMBAKAU' tidak ditemukan dalam data.")
        with col2:
            st.write("**Penjelasan Grafik:**")
            st.info("""
            Grafik garis ini menunjukkan tren pengeluaran rokok. 
            **Hover** di atas garis atau titik untuk melihat nilai pasti per tahun.
            """)

    st.write("Tren Tahunan per Daerah")
    with st.expander("Klik untuk melihat detail"):
        # Memastikan df_final ada sebelum membuat selectbox
        st.write("Pilih satu Kabupaten/Kota untuk melihat perkembangan pengeluaran rokok, IPM, dan tingkat kemiskinan dari tahun ke tahun.")
        if 'df_final' in st.session_state and df_final is not None:
            # 1. Buat daftar daerah unik untuk pilihan
            list_daerah = sorted([daerah for daerah in df_final['Kabupaten/Kota'].unique() if isinstance(daerah, str)])

            if not list_daerah:
                st.warning("Tidak ada nama daerah yang valid ditemukan dalam data hasil analisis.")
            else:
                # 2. Buat Selectbox untuk memilih daerah
                selected_daerah = st.selectbox(
                    "Pilih Kabupaten/Kota:",
                    options=['-- Pilih Daerah --'] + list_daerah, # Tambah opsi default
                    index=0 # Default ke '-- Pilih Daerah --'
                )

                # 3. Proses jika daerah sudah dipilih
                if selected_daerah != '-- Pilih Daerah --':
                    st.subheader(f"Tren Tahunan untuk: {selected_daerah}")

                    # 4. Filter data HANYA untuk daerah yang dipilih
                    df_daerah = df_final[df_final['Kabupaten/Kota'] == selected_daerah].sort_values('Tahun')

                    if df_daerah.empty:
                        st.warning(f"Tidak ditemukan data untuk {selected_daerah} pada rentang tahun terpilih.")
                    else:
                        # Ambil label klaster (semua tahun sama untuk daerah ini)
                        # Memastikan ada data sebelum mengambil iloc[0]
                        if not df_daerah.empty:
                            daerah_cluster_label = df_daerah['Cluster_Label'].iloc[0]
                            # Ambil warna dari master map, default abu-abu jika tidak ada
                            daerah_color = master_color_map.get(daerah_cluster_label, 'grey')
                            st.info(f"Daerah ini termasuk dalam: **{daerah_cluster_label}**") # Tampilkan info klaster
                        else:
                            daerah_cluster_label = "N/A"
                            daerah_color = 'grey'

                        # 5. Daftar fitur yang akan diplot
                        fitur_rokok_plot = results.get('selected_features_run', [])
                        fitur_tambahan = []
                        if 'IPM' in df_daerah.columns: fitur_tambahan.append('IPM')
                        if 'Persentase_Miskin' in df_daerah.columns: fitur_tambahan.append('Persentase_Miskin')
                        all_fitur_to_plot = fitur_rokok_plot + fitur_tambahan

                        if not all_fitur_to_plot:
                            st.warning("Tidak ada fitur yang tersedia untuk diplot.")
                        else:
                            # 6. Buat layout kolom
                            num_cols = 1
                            cols = st.columns(num_cols)
                            col_index = 0

                            # 7. Loop melalui fitur dan buat plot
                            for fitur in all_fitur_to_plot:
                                if fitur in df_daerah.columns:
                                    with cols[col_index]:
                                        fig_line_daerah = px.line(
                                            df_daerah,
                                            x='Tahun',
                                            y=fitur,
                                            title=f"{fitur}",
                                            markers=True,
                                            height=300,
                                            color_discrete_sequence=[daerah_color],
                                            custom_data=['Cluster', 'Provinsi', 'Kabupaten/Kota']
                                        )

                                        # Format hover template
                                        hover_label_fitur = "Nilai:"
                                        if fitur == 'Persentase_Miskin': hover_label_fitur = "Kemiskinan (%):"
                                        elif fitur == 'IPM': hover_label_fitur = "IPM:"
                                        elif 'Rokok' in fitur or 'Tembakau' in fitur : hover_label_fitur = "Pengeluaran:"

                                        fig_line_daerah.update_traces(
                                            hovertemplate=
                                                f"<b>Klaster:</b> %{{customdata[0]}}<br>" +
                                                f"<b>Provinsi:</b> %{{customdata[1]}}<br>" +
                                                f"<b>Kab/Kota:</b> %{{customdata[2]}}<br>" +
                                                f"<b>Tahun:</b> %{{x}}<br>" +
                                                f"<b>{hover_label_fitur}</b> %{{y:,.2f}}" +
                                                "<extra></extra>" # Sembunyikan info trace tambahan
                                        )

                                        fig_line_daerah.update_layout(
                                            margin=dict(l=20, r=20, t=40, b=20),
                                            xaxis_title=None,
                                            yaxis_title=None,
                                            dragmode='pan'
                                        )
                                        st.plotly_chart(fig_line_daerah, use_container_width=True)

                                    col_index = (col_index + 1) % num_cols
                                else:
                                    with cols[col_index]:
                                        st.warning(f"Kolom '{fitur}' tidak ditemukan.")
                                    col_index = (col_index + 1) % num_cols
        else:
            st.info("Jalankan proses klasterisasi terlebih dahulu untuk dapat melihat tren per daerah.")

    #Top 10 Barchart
    st.write("**Top 10 Daerah dengan Pengeluaran Tertinggi**")
    with st.expander("Klik untuk melihat detail"):
        
        if not df_final.empty:
            for year in sorted(df_final['Tahun'].unique()):
                with st.expander(f"Tahun {year}"):
                    df_year = df_final[df_final['Tahun'] == year]
                    for feature in features_run:
                        st.write(f"**{feature}**")
                        # Ambil top 10 dan urutkan
                        top_10_df = df_year.nlargest(10, feature).sort_values(feature, ascending=False)
                        
                        if 'Provinsi' not in top_10_df.columns:
                                st.warning("Kolom 'Provinsi' tidak ditemukan untuk hover.")
                                hover_cols = {feature: ':.2f'} # Data hover default
                        else:
                                # Data hover termasuk Provinsi
                                hover_cols = {'Provinsi': True, feature: ':.2f'}

                        fig_bar_top10 = px.bar(
                                top_10_df,
                                x='Kabupaten/Kota',
                                y=feature,
                                color='Cluster_Label',
                                color_discrete_map=master_color_map,
                                title=f'Top 10 Pengeluaran - {feature} ({year})',
                                hover_name='Kabupaten/Kota',
                                hover_data=hover_cols
                            )

                        fig_bar_top10.update_layout(xaxis_title="Kabupaten/Kota", yaxis_title="Nilai Pengeluaran", dragmode='pan')
                        st.plotly_chart(fig_bar_top10, use_container_width=True)


        
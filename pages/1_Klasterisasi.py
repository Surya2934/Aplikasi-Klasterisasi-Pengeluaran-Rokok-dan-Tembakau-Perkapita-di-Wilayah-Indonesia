import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances_argmin_min, silhouette_samples
from sklearn.decomposition import PCA
import time
import numpy as np
import geopandas as gpd 
import io
import matplotlib.cm as cm
import plotly.express as px
import folium
from streamlit_folium import st_folium

# --- Import kode K-Means++ Manual ---
try:
    from k_means_plus_plus import KMeansPlusPlusManual
except ImportError:
    st.error("File 'k_means_plus_plus.py' tidak ditemukan.")
    KMeansPlusPlusManual = None

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Utama", layout="wide")

# --- FUNGSI CALLBACK UNTUK MEMBERSIHKAN STATE ---
def clear_analysis_results():
    """Hapus hasil analisis sebelumnya dari session state saat sumber data diubah."""
    # Kunci lama: 'df_final', 'results_info', 'df_display_final'
    # Kunci baru: 'analysis_results_list'
    keys_to_clear = ['analysis_results_list', 'eval_graph_data', 'df_analysis'] 
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # st.toast("Sumber data diubah, hasil analisis sebelumnya dibersihkan.")

# --- FUNGSI-FUNGSI UTAMA ---
@st.cache_data
def load_data(file_path):
    df = None # Inisialisasi df
    is_uploaded_file = hasattr(file_path, 'seek') # Cek apakah ini file unggahan

    try:
        # --- 1. Handle File Unggahan ---
        if is_uploaded_file:
            filename = file_path.name.lower()
            file_path.seek(0) 

            if filename.endswith('.xlsx'):
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                    st.info("Membaca file sebagai Excel (.xlsx).")
                except Exception as e_excel:
                    st.error(f"Gagal membaca file '{filename}' sebagai Excel. Error: {e_excel}")
                    file_path.seek(0)
                    pass # Biarkan coba CSV
            
            # Jika bukan Excel atau Excel gagal, coba CSV
            if df is None and (filename.endswith('.csv') or not filename.endswith('.xlsx')):
                # st.info("Mencoba membaca file CSV unggahan...")
                # Untuk file UPLOAD, coba Semicolon (;) dulu berdasarkan kasus Anda
                try: 
                    df = pd.read_csv(file_path, delimiter=';')
                    st.success("Berhasil membaca CSV (upload) dengan pemisah ';'.")
                except Exception:
                    file_path.seek(0) 
                    try: # Fallback ke Koma (,)
                        df = pd.read_csv(file_path, delimiter=',') 
                        st.success("Berhasil membaca CSV (upload) dengan pemisah ','.")
                    except Exception as e_comma:
                        st.error(f"Gagal membaca file CSV (upload) dengan ';' atau ','. Error: {e_comma}")
                        return None 

        # --- 2. Handle File Path (Dataset Bawaan) ---
        else: 
            filename = str(file_path).lower()
            if filename.endswith('.csv'):
                 # --- PERBAIKAN: Coba KOMA (,) dulu untuk data bawaan ---
                 # st.info("Membaca dataset bawaan...")
                 try:
                    df = pd.read_csv(filename, delimiter=',') # Coba Koma dulu (standar)
                    # st.success("Berhasil membaca dataset bawaan dengan ','.")
                 except Exception:
                     # st.warning(f"Gagal membaca '{filename}' dengan ','. Mencoba ';'.")
                     try:
                        df = pd.read_csv(filename, delimiter=';') # Fallback ke Semicolon
                        # st.success("Berhasil membaca dataset bawaan dengan ';'.")
                     except Exception as e_csv:
                        st.error(f"Gagal membaca file CSV bawaan '{filename}' dengan ',' atau ';'. Error: {e_csv}")
                        return None
                 # --- AKHIR PERBAIKAN ---
            elif filename.endswith('.xlsx'):
                 try:
                    df = pd.read_excel(filename, engine='openpyxl')
                 except Exception as e_excel:
                    st.error(f"Gagal membaca file '{filename}' sebagai Excel. Error: {e_excel}")
                    return None
            else:
                 st.error(f"Format file bawaan '{filename}' tidak didukung.")
                 return None

        # --- 3. Post-processing ---
        if df is not None:
            # Hapus info debug jika sudah tidak perlu
            # st.write("--- Debug Info: Kolom Terdeteksi ---")
            # st.code(f"{df.columns.tolist()}")
            # st.write("--- Akhir Debug Info ---")
             
            if 'Kabupaten/Kota' in df.columns and 'Tahun' in df.columns:
                df.drop_duplicates(subset=['Kabupaten/Kota', 'Tahun'], keep='first', inplace=True)
            return df
        else:
             st.error("Gagal memuat data dari file.")
             return None

    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error: Gagal memproses file. Detail: {e}")
        return None

@st.cache_data
def load_geodata(file_path):
    """Memuat file GeoJSON dengan penanganan CRS."""
    try:
        gdf = gpd.read_file(file_path)
        # Penanganan CRS untuk Folium
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        st.error(f"Error: Gagal memuat file GeoJSON '{file_path}'. Detail: {e}")
        return None

def run_clustering(df_aggregated, selected_features, algorithm_choice, params):
    """Menjalankan algoritma klasterisasi pada data agregat."""
    if not selected_features:
        st.error("Harap pilih setidaknya satu fitur (kesalahan internal di run_clustering).")
        return {}
    
    df_cluster = df_aggregated.copy()
    df_cluster[selected_features] = df_cluster[selected_features].fillna(0)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_cluster[selected_features])

    model = None
    model_params = {} # Simpan parameter yg digunakan

    try:
        if algorithm_choice == 'K-Means++':
            if KMeansPlusPlusManual:
                k_val = params.get('k', 3) # Ambil K dari params
                model_params = {'n_clusters': k_val, 'n_init': 10}
                model = KMeansPlusPlusManual(n_clusters=k_val, random_state=42, n_init=10)
        elif algorithm_choice == 'K-Means':
            k_val = params.get('k', 3)
            model_params = {'n_clusters': k_val, 'n_init': 10, 'init': 'random'}
            model = KMeans(n_clusters=k_val, init='random', n_init=10, random_state=42)
        elif algorithm_choice == 'OPTICS':
            min_samples_val = params.get('min_samples', 5) # Ambil min_samples
            model_params = {'min_samples': min_samples_val, 'xi': 0.05, 'min_cluster_size': 5}
            model = OPTICS(min_samples=min_samples_val, xi=0.05, min_cluster_size=5)
    except Exception as e:
        st.error(f"Error saat inisialisasi model {algorithm_choice}: {e}")
        return {}

    if model is None:
        st.error(f"Gagal menginisialisasi model untuk {algorithm_choice}.")
        return {}
        
    df_cluster['Cluster'] = model.fit_predict(X_scaled)
    
    labels = df_cluster['Cluster']
    results = {
        'df_result': df_cluster, # Ini adalah df_aggregated + kolom Cluster
        'X_scaled': X_scaled,    # Ini adalah X_scaled_agg
        'model': model,
        'model_params_used': model_params,
        'algo_run': algorithm_choice,
        'selected_features_run': selected_features
    }

    # Hitung metrik (hanya untuk data non-noise)
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

# --- FUNGSI TAMPILKAN HASIL (BARU) ---
def display_analysis_results(result_package):
    """
    Fungsi terpusat untuk menampilkan SEMUA visualisasi dan tabel
    untuk SATU hasil analisis.
    """
    try:
        # 1. Ekstrak data dari paket
        results = result_package['results_info']
        df_final = result_package['df_final_labeled']
        algo_name = result_package['algo_name']
        features_run = results.get('selected_features_run', []) # Fitur yg dipakai
        df_result_agg = results.get('df_result') # Data agregat + klaster
        X_scaled_agg = results.get('X_scaled')   # Data agregat scaled
        
        # --- 2. Definisikan Skema Warna & Label Terpusat (PERBAIKAN) ---
        df_non_noise = df_final[df_final['Cluster'] != -1]
        unique_clusters = sorted(df_non_noise['Cluster'].unique())
        colors_sns = sns.color_palette('viridis', n_colors=len(unique_clusters))
        colors_hex = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for (r,g,b) in colors_sns]

        # Buat map label (Numeric -> String)
        master_label_map = {cluster: f"Klaster {cluster}" for cluster in unique_clusters}
        master_label_map[-1] = "Noise"

        # --- PERBAIKAN: Buat master_color_map DENGAN KUNCI STRING ---
        # Ini penting agar cocok dengan 'Cluster_Label'
        master_color_map = {}
        for numeric_key, string_label in master_label_map.items():
            if numeric_key == -1:
                master_color_map[string_label] = 'grey' # Warna noise
            elif numeric_key in unique_clusters:
                try:
                    idx = unique_clusters.index(numeric_key)
                    master_color_map[string_label] = colors_hex[idx] # Warna klaster
                except ValueError:
                    master_color_map[string_label] = 'black' # Fallback
        
        master_color_map['Tidak Ada Data'] = 'lightgrey' # Warna untuk yg tdk punya data
        # --- AKHIR PERBAIKAN MAP WARNA ---
        
        # Label sudah dibuat saat proses, tapi pastikan lagi
        if 'Cluster_Label' not in df_final.columns:
             df_final['Cluster_Label'] = df_final['Cluster'].map(master_label_map)
        if 'Cluster_Label' not in df_result_agg.columns:
             df_result_agg['Cluster_Label'] = df_result_agg['Cluster'].map(master_label_map)


        # --- 3. Tampilkan Metrik ---
        st.subheader("Metrik Evaluasi")
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{results.get('sil', 0):.4f}" if 'sil' in results else "N/A")
        col2.metric("Davies-Bouldin Index", f"{results.get('dbi', 0):.4f}" if 'dbi' in results else "N/A")
        col3.metric("Waktu Proses", f"{results.get('runtime', 0):.2f} detik")

        if 'n_clusters_found' in results:
            col1b, col2b = st.columns(2)
            col1b.metric("Jumlah Klaster Ditemukan", results.get('n_clusters_found', 0))
            col2b.metric("Jumlah Data Noise", results.get('n_noise', 0))
        
        # --- 4. Tampilkan Detail Klaster (Expander) ---
        st.subheader("Informasi Detail Klaster")
        with st.expander("Klik untuk melihat detail statistik dan teknis klaster"):
            st.write("**Statistik Rata-rata per Klaster (Data Asli):**")
            
            # Buat dict agregasi (dari data asli df_final)
            agg_dict_detail = {'ROKOK DAN TEMBAKAU': 'mean', 'Rokok kretek filter' : 'mean', 'Rokok kretek tanpa filter' : 'mean', 'Rokok putih' : 'mean', 'Tembakau' : 'mean', 'Rokok dan tembakau Lainnya' : 'mean'}
            rename_dict = {'ROKOK DAN TEMBAKAU': 'Rata-rata Total Pengeluaran Rokok'}
            if 'IPM' in df_final.columns: agg_dict_detail['IPM'] = 'mean'; rename_dict['IPM'] = 'Rata-rata IPM'
            if 'Persentase_Miskin' in df_final.columns: agg_dict_detail['Persentase_Miskin'] = 'mean'; rename_dict['Persentase_Miskin'] = 'Rata-rata Kemiskinan (%)'

            df_profiling = df_final[df_final['Cluster'] != -1] # Filter noise
            if not df_profiling.empty:
                # Filter agg_dict_detail agar hanya berisi kolom yg ada di df_profiling
                valid_agg_keys = {k: v for k, v in agg_dict_detail.items() if k in df_profiling.columns}
                if valid_agg_keys:
                    profiling_data = df_profiling.groupby('Cluster').agg(valid_agg_keys).rename(columns=rename_dict)
                    st.dataframe(profiling_data.style.format("{:.2f}"))
                else:
                    st.write("Kolom fitur tidak ditemukan untuk profiling.")
            else:
                st.write("Tidak ada klaster valid (non-noise) untuk dianalisis.")

            st.write("**Informasi Teknis Model (Data Agregat):**")
            model = results.get('model')
            if hasattr(model, 'n_iter_'):
                st.write(f"- Jumlah Iterasi yang Dibutuhkan: **{model.n_iter_}**")

            st.write("- Posisi Centroid Terakhir (pada data agregat yang diskalakan):")
            if hasattr(model, 'centroids'): # K-Means++ Manual
                centroids_final = model.centroids
                st.dataframe(pd.DataFrame(centroids_final, columns=features_run).style.format("{:.4f}"))
            elif hasattr(model, 'cluster_centers_'): # K-Means Sklearn
                centroids_final = model.cluster_centers_
                st.dataframe(pd.DataFrame(centroids_final, columns=features_run).style.format("{:.4f}"))
            elif algo_name == 'OPTICS':
                st.write("OPTICS tidak menghasilkan centroid tetap.")
        
        st.subheader(f"Visualisasi Hasil")
        
        # --- 5. Visualisasi (Semua Plot) ---
        
        # REACHABILITY PLOT (KHUSUS OPTICS)
        if 'n_clusters_found' in results:
            st.write("**Grafik Reachability OPTICS**")
            with st.expander("Klik untuk melihat detail"):
                col1_r, col2_r = st.columns([2, 1])
                with col1_r:
                    if model:
                        reachability = model.reachability_[model.ordering_]
                        labels_plot = model.labels_[model.ordering_]
                        space = np.arange(len(reachability))
                        df_reach = pd.DataFrame({'Sample Order': space, 'Reachability Distance': reachability, 'Cluster': labels_plot})
                        df_reach['Cluster_Label'] = df_reach['Cluster'].map(master_label_map)

                        fig_reach = px.bar(df_reach, x='Sample Order', y='Reachability Distance', color='Cluster_Label', color_discrete_map=master_color_map, title='OPTICS Reachability Plot')
                        fig_reach.update_layout(bargap=0, showlegend=False, dragmode='pan') 
                        st.plotly_chart(fig_reach, use_container_width=True)
                with col2_r:
                    st.write("**Penjelasan Grafik:**")
                    st.info("Grafik ini menunjukkan 'kepadatan' data. Lembah (area rendah) menunjukkan klaster padat, Puncak memisahkan klaster, dan Bar Abu-abu adalah noise.")

        # Tabel Data
        st.write(f"Tabel Data Hasil Klasterisasi")
        with st.expander("Klik untuk melihat detail"):
            display_df = df_final.copy() # Gunakan df_final dari paket
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                if 'Provinsi' in display_df.columns:
                    provinsi_list = [prov for prov in display_df['Provinsi'].unique().tolist() if isinstance(prov, str)]
                    all_provinces = ['Semua Provinsi'] + sorted(provinsi_list)
                    selected_province = st.selectbox("Filter Provinsi:", options=all_provinces, key=f"prov_select_{algo_name}") # Key unik
            with filter_col2:
                search_term = st.text_input("Cari Kabupaten/Kota:", key=f"search_{algo_name}")
            with filter_col3:
                if 'Cluster_Label' in display_df.columns:
                    unique_labels = display_df['Cluster_Label'].unique().tolist()
                    available_clusters = sorted([str(label) for label in unique_labels])
                    if available_clusters:
                        selected_clusters = st.multiselect("Filter Klaster:", options=available_clusters, default=available_clusters, key=f"cluster_select_{algo_name}")
            
            # Terapkan filter
            if 'selected_province' in locals() and selected_province != 'Semua Provinsi':
                display_df = display_df[display_df['Provinsi'] == selected_province]
            if search_term:
                display_df = display_df[display_df['Kabupaten/Kota'].str.contains(search_term, case=False, na=False)]
            if 'selected_clusters' in locals() and len(selected_clusters) != len(available_clusters):
                 display_df = display_df[display_df['Cluster_Label'].isin(selected_clusters)]

            # Sembunyikan kolom bantu
            columns_to_hide = ['Cluster_Label', 'key_df', 'id_key', 'merge_key', 'key_geojson']
            cols_exist_to_hide = [col for col in columns_to_hide if col in display_df.columns]
            df_to_display = display_df.drop(columns=cols_exist_to_hide) if cols_exist_to_hide else display_df
            df_to_display = df_to_display.reset_index(drop=True) # Reset index
            
            st.dataframe(df_to_display)

        # Distribusi Klaster (Agregat)
        st.write("**Distribusi Anggota per Klaster (Agregat)**")
        with st.expander("Klik untuk melihat detail"):
            col1_dist, col2_dist = st.columns([2, 1])
            with col1_dist:
                if df_result_agg is not None:
                    counts_df = df_result_agg.groupby('Cluster_Label').size().reset_index(name='Jumlah')
                    fig1 = px.bar(counts_df, x='Cluster_Label', y='Jumlah', color='Cluster_Label', color_discrete_map=master_color_map, title="Distribusi Anggota per Klaster")
                    fig1.update_layout(xaxis_title="Klaster", yaxis_title="Jumlah Kabupaten/Kota", dragmode='pan')
                    st.plotly_chart(fig1, use_container_width=True)
            with col2_dist:
                st.info("Grafik ini menunjukkan jumlah Kabupaten/Kota per klaster (agregat).")
                if df_result_agg is not None:
                    cluster_counts = df_result_agg['Cluster_Label'].value_counts().sort_index()
                    for label, count in cluster_counts.items():
                        st.write(f"- {label}: **{count}** anggota (Kab/Kota)")

        # Scatter Plot / Strip Plot (Persebaran Klaster Agregat)
        st.write("**Visualisasi Persebaran Klaster (Agregat)**")
        with st.expander("Klik untuk melihat detail"):
            col1_pca, col2_pca = st.columns([2, 1])
            with col1_pca:
                if X_scaled_agg is not None and df_result_agg is not None and features_run:
                    if len(features_run) > 1: # PCA Plot
                        required_cols_pca = ['Kabupaten/Kota', 'Cluster_Label']
                        if all(col in df_result_agg.columns for col in required_cols_pca):
                            pca = PCA(n_components=2)
                            principal_components = pca.fit_transform(X_scaled_agg)
                            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                            pca_df['Cluster_Label'] = df_result_agg['Cluster_Label'].values
                            pca_df['Kabupaten/Kota'] = df_result_agg['Kabupaten/Kota'].values
                            fig3 = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster_Label', color_discrete_map=master_color_map, title='Persebaran Klaster (PCA)', custom_data=['Kabupaten/Kota', 'Cluster_Label'])
                            fig3.update_traces(hovertemplate="<b>Kab/Kota:</b> %{customdata[0]}<br><b>Klaster:</b> %{customdata[1]}<extra></extra>")
                            fig3.update_layout(dragmode='pan', legend_title_text='Klaster')
                            st.plotly_chart(fig3, use_container_width=True)
                        else: st.warning(f"Kolom hover untuk PCA tidak ditemukan. Membutuhkan: {required_cols_pca}")
                    
                    elif len(features_run) == 1: # Strip Plot
                        single_feature_name = features_run[0]
                        st.info(f"Menampilkan Strip Plot (1D) untuk: {single_feature_name}")
                        required_cols_strip = [single_feature_name, 'Kabupaten/Kota', 'Cluster_Label']
                        if all(col in df_result_agg.columns for col in required_cols_strip):
                            fig_strip = px.strip(df_result_agg, x='Cluster_Label', y=single_feature_name, color='Cluster_Label', color_discrete_map=master_color_map, title=f'Persebaran Klaster: {single_feature_name}', custom_data=['Kabupaten/Kota', 'Cluster_Label', single_feature_name])
                            fig_strip.update_traces(hovertemplate="<b>Kab/Kota:</b> %{customdata[0]}<br><b>Klaster:</b> %{customdata[1]}<br><b>{single_feature_name}:</b> %{{customdata[2]:.2f}}<extra></extra>")
                            fig_strip.update_layout(dragmode='pan', xaxis_title="Klaster")
                            st.plotly_chart(fig_strip, use_container_width=True)
                        else: st.warning(f"Kolom untuk Strip Plot tidak ditemukan. Membutuhkan: {required_cols_strip}")
            with col2_pca:
                st.info("Visualisasi persebaran data agregat (median) per Kab/Kota. Jika 2+ fitur, menggunakan PCA. Jika 1 fitur, menggunakan Strip Plot.")

        # Box Plot (Profil Klaster)
        st.write("**Profil Karakteristik Klaster (Data Asli per Tahun)**")
        with st.expander("Klik untuk melihat detail"):
            st.write("**Distribusi Pengeluaran per Jenis Rokok (Termasuk Noise)**")
            col1_box, col2_box = st.columns([2, 1])
            with col1_box:
                if not df_final.empty:
                    df_melted_all = df_final.melt(id_vars=['Cluster_Label'], value_vars=features_run, var_name='Jenis Rokok', value_name='Nilai Pengeluaran')
                    fig2_all = px.box(df_melted_all, x='Jenis Rokok', y='Nilai Pengeluaran', color='Cluster_Label', color_discrete_map=master_color_map, title="Profil Pengeluaran Rokok (Termasuk Noise)", points='outliers')
                    fig2_all.update_layout(legend_title_text='Klaster', dragmode='pan')
                    fig2_all.update_xaxes(tickangle=45)
                    st.plotly_chart(fig2_all, use_container_width=True)
            with col2_box:
                st.write("**Cara Membaca Boxplot:**")
            st.markdown("---")
            st.write("**Perbandingan IPM & Kemiskinan (Termasuk Noise)**")
            col1_box_extra, col2_box_extra = st.columns(2)
            with col1_box_extra:
                if 'IPM' in df_final.columns:
                    fig_ipm_all = px.box(df_final, x='Cluster_Label', y='IPM', color='Cluster_Label', color_discrete_map=master_color_map, title="Boxplot IPM (Termasuk Noise)", points='outliers')
                    fig_ipm_all.update_layout(xaxis_title="Klaster/Noise", showlegend=False, dragmode='pan')
                    st.plotly_chart(fig_ipm_all, use_container_width=True)
            with col2_box_extra:
                if 'Persentase_Miskin' in df_final.columns:
                    fig_pov_all = px.box(df_final, x='Cluster_Label', y='Persentase_Miskin', color='Cluster_Label', color_discrete_map=master_color_map, title="Boxplot Kemiskinan (Termasuk Noise)", points='outliers')
                    fig_pov_all.update_layout(xaxis_title="Klaster/Noise", showlegend=False, dragmode='pan')
                    st.plotly_chart(fig_pov_all, use_container_width=True)

        st.write("**Pemetaan Geografis Hasil Klasterisasi**")
        with st.expander("Klik untuk melihat detail"):
            min_map_year = int(df_final['Tahun'].min())
            max_map_year = int(df_final['Tahun'].max())
            map_year = min_map_year
            if min_map_year < max_map_year:
                map_year = st.slider("Pilih Tahun Peta:", min_map_year, max_map_year, max_map_year, key=f"map_slider_{algo_name}")
            
            with st.spinner(f"Memuat peta Folium untuk {algo_name}..."):
                geojson_path = 'Peta Indonesia Kota Kabupaten simplified.json'
                peta_indo_gdf = load_geodata(geojson_path)
                
                if peta_indo_gdf is not None:
                    # Siapkan data untuk peta
                    df_peta = df_final[df_final['Tahun'] == map_year].copy()
                    if 'key_geojson' not in peta_indo_gdf.columns:
                        peta_indo_gdf['key_geojson'] = peta_indo_gdf['NAME_2'].str.upper()
                    if 'key_df' not in df_peta.columns:
                        df_peta['key_df'] = df_peta['Kabupaten/Kota'].str.upper()
                    
                    df_peta['Cluster_Label'] = df_peta['Cluster_Label'].fillna('Tidak Ada Data').astype(str)

                    # --- PERBAIKAN: Tambahkan Provinsi & Rokok Total ke Tooltip ---
                    tooltip_fields = ['Kabupaten/Kota', 'Cluster_Label']
                    tooltip_aliases = ['Kab/Kota:', 'Klaster:']
                    
                    # Fungsi helper untuk format string angka
                    def safe_to_str_format(series, decimals=2, default_val="N/A", as_int_comma=False):
                        try:
                            numeric_series = pd.to_numeric(series, errors='coerce')
                            if as_int_comma:
                                return numeric_series.apply(lambda x: f"{int(x):,}" if pd.notna(x) else default_val).fillna(default_val)
                            else:
                                return numeric_series.round(decimals).astype(str).fillna(default_val)
                        except Exception: 
                             return pd.Series([default_val] * len(series), index=series.index)
                    
                    # Tambahkan Provinsi (jika ada)
                    if 'Provinsi' in df_peta.columns:
                        tooltip_fields.append('Provinsi')
                        tooltip_aliases.append('Provinsi:')
                    
                    # Tambahkan ROKOK DAN TEMBAKAU (jika ada)
                    if 'ROKOK DAN TEMBAKAU' in df_peta.columns:
                        df_peta['Rokok_Total_str'] = safe_to_str_format(df_peta['ROKOK DAN TEMBAKAU'], as_int_comma=True) # Format sbg integer
                        tooltip_fields.append('Rokok_Total_str')
                        tooltip_aliases.append('Pengeluaran Rokok Total:')

                    if 'IPM' in df_peta.columns:
                        df_peta['IPM_str'] = safe_to_str_format(df_peta['IPM'])
                        tooltip_fields.append('IPM_str'); tooltip_aliases.append('IPM:')
                    if 'Persentase_Miskin' in df_peta.columns:
                        df_peta['Miskin_str'] = safe_to_str_format(df_peta['Persentase_Miskin'])
                        tooltip_fields.append('Miskin_str'); tooltip_aliases.append('Kemiskinan (%):')
                    # --- AKHIR PERBAIKAN TOOLTIP ---

                    # Gabungkan GDF dengan data
                    merged_gdf = peta_indo_gdf.merge(df_peta, left_on='key_geojson', right_on='key_df', how='left')
                    merged_gdf['Cluster_Label'] = merged_gdf['Cluster_Label'].fillna('Tidak Ada Data').astype(str)
                    merged_gdf['Kabupaten/Kota'] = merged_gdf['Kabupaten/Kota'].fillna(merged_gdf['NAME_2'])
                    for col in tooltip_fields:
                        if col not in merged_gdf.columns: merged_gdf[col] = 'N/A'
                        elif '_str' in col or col == 'Provinsi': merged_gdf[col] = merged_gdf[col].fillna('N/A')

                    # Buat peta
                    m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles='cartodbpositron')
                    
                    # Buat style function (ini akan mewarnai peta)
                    def style_function(feature):
                        label = 'Tidak Ada Data'
                        properties = feature.get('properties', {})
                        if properties:
                            label = properties.get('Cluster_Label', 'Tidak Ada Data')
                        # Ambil warna dari map yg KUNCI-nya STRING
                        fill_color = master_color_map.get(label, '#D3D3D3')
                        return {'fillColor': fill_color, 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.7}

                    # Tambahkan GeoJson (warna + tooltip)
                    folium.GeoJson(
                        merged_gdf,
                        style_function=style_function,
                        tooltip=folium.features.GeoJsonTooltip(
                            fields=tooltip_fields, # <-- List tooltip yang sudah diupdate
                            aliases=tooltip_aliases, # <-- List alias yang sudah diupdate
                            localize=True, sticky=False, labels=True,
                            style="background-color: #F0EFEF; border: 2px solid black; border-radius: 3px; box-shadow: 3px;",
                            max_width=800,
                        ),
                        highlight_function=lambda x: {'weight':3, 'fillOpacity':1},
                        name=f'Peta Klaster {map_year}'
                    ).add_to(m)
                    
                    # --- PERBAIKAN: Tambahkan Legenda Manual (Versi Stabil) ---
                    legend_html = f'''
                         <div style="position: absolute; 
                         bottom: 50px; left: 50px; width: auto; height: auto; 
                         background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
                         border-radius: 6px; padding: 10px;
                         color: black; 
                         ">
                         <b>Legenda Klaster</b><br style="margin-bottom: 5px;">
                      '''
                    
                    #Logika sorting
                    sorted_items = sorted(master_color_map.items(), key=lambda item: (
                        'Z' if item[0] in ['Tidak Ada Data', 'Noise'] else item[0], item[0]
                    ))
                    
                    #Loop untuk menambahkan baris legenda
                    for label, color in sorted_items:
                        opacity = 0.3 if label == 'Tidak Ada Data' else 0.7
                        
                        #Tambahkan baris
                        legend_html += '<div style="display: flex; align-items: center; margin-bottom: 4px;">'
                        
                        #Tambahkan kotak warna
                        legend_html += f'<i style="background: {color}; width: 18px; height: 18px; margin-right: 8px; border: 1px solid black; opacity: {opacity};"></i>'
                        
                        #Tambahkan label teks
                        legend_html += f'<span>{label}</span>'
                        
                        #Tutup baris
                        legend_html += '</div>'
                    
                    #Tutup div utama legenda
                    legend_html += '</div>'

                    #Tambahkan elemen HTML ke peta
                    m.get_root().html.add_child(folium.Element(legend_html))

                    folium.LayerControl().add_to(m)
                    st_folium(m, width=725, height=500, key=f"map_display_{algo_name}") # Key unik
                else:
                    st.error("Gagal memuat file GeoJSON untuk peta.")

        # Tren Pengeluaran Rokok (Line Plot)
        st.write("**Perkembangan Rata-rata Total Pengeluaran Rokok per Klaster**")
        with st.expander("Klik untuk melihat detail"):
            col1_trend, col2_trend = st.columns([2, 1])
            with col1_trend:
                if 'ROKOK DAN TEMBAKAU' in df_final.columns:
                    df_no_noise = df_final[df_final['Cluster'] != -1]
                    if not df_no_noise.empty:
                        avg_expenditure = df_no_noise.groupby(['Cluster_Label', 'Tahun'])['ROKOK DAN TEMBAKAU'].mean().reset_index()
                        fig_line = px.line(avg_expenditure, x='Tahun', y='ROKOK DAN TEMBAKAU', color='Cluster_Label', color_discrete_map=master_color_map, markers=True, title='Rata-rata Total Pengeluaran Rokok per Klaster per Tahun')
                        fig_line.update_layout(yaxis_title='Rata-rata Pengeluaran', legend_title_text='Klaster', dragmode='pan')
                        st.plotly_chart(fig_line, use_container_width=True)
                    else: st.info("Tidak ada klaster valid untuk menampilkan tren.")
            with col2_trend:
                st.info("Grafik ini menunjukkan tren perubahan rata-rata pengeluaran rokok total per klaster (non-noise) dari tahun ke tahun.")

        # Top 10 Barchart
        st.write("**Top 10 Daerah dengan Pengeluaran Tertinggi**")
        with st.expander("Klik untuk melihat detail"):
            if not df_final.empty:
                # 'algo_name' sudah tersedia sebagai parameter di fungsi display_analysis_results
                
                for year in sorted(df_final['Tahun'].unique()):
                    with st.expander(f"Tahun {year}"):
                        df_year = df_final[df_final['Tahun'] == year]
                        for feature in features_run:
                            st.write(f"**{feature}**")
                            top_10_df = df_year.nlargest(10, feature).sort_values(feature, ascending=False)
                            if not top_10_df.empty:
                                # Siapkan hover_cols (tanpa Cluster_Label, sesuai permintaan sebelumnya)
                                hover_cols = {}
                                if 'Provinsi' in top_10_df.columns:
                                    hover_cols['Provinsi'] = True
                                hover_cols[feature] = ':.2f' # Pastikan nilai fitur ada
                                
                                fig_bar_top10 = px.bar(
                                    top_10_df, 
                                    x='Kabupaten/Kota', 
                                    y=feature, 
                                    color='Cluster_Label', 
                                    color_discrete_map=master_color_map, 
                                    title=f'Top 10 - {feature} ({year})', 
                                    hover_name='Kabupaten/Kota', 
                                    hover_data=hover_cols
                                )
                                fig_bar_top10.update_layout(xaxis_title="Kabupaten/Kota", yaxis_title="Nilai Pengeluaran", dragmode='pan')
                                
                                # --- PERBAIKAN: Tambahkan key unik ---
                                # Key unik = nama_algoritma + tahun + nama_fitur
                                unique_key = f"top10_plot_{algo_name}_{year}_{feature}"
                                st.plotly_chart(fig_bar_top10, use_container_width=True, key=unique_key)
                                # --- AKHIR PERBAIKAN ---

        # Silhouette Plot (Matplotlib)
        st.write("**Visualisasi Silhouette per Klaster (Agregat)**")
        with st.expander("Klik untuk melihat detail"):
            col1_sil, col2_sil = st.columns([2,1])
            with col1_sil:
                if X_scaled_agg is not None and df_result_agg is not None:
                    labels = df_result_agg['Cluster'].values
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    avg_silhouette_score = results.get('sil')
                    if n_clusters > 1 and avg_silhouette_score is not None:
                        fig_sil, ax_sil = plt.subplots(1, 1, figsize=(7, 5))
                        y_lower = 10
                        sample_silhouette_values = silhouette_samples(X_scaled_agg, labels)
                        for i in range(n_clusters):
                            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
                            ith_cluster_silhouette_values.sort()
                            size_cluster_i = ith_cluster_silhouette_values.shape[0]
                            y_upper = y_lower + size_cluster_i
                            color = cm.viridis(float(i) / n_clusters)
                            ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
                            ax_sil.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                            y_lower = y_upper + 10
                        ax_sil.set_title("Silhouette Plot per Klaster (Agregat)")
                        ax_sil.set_xlabel("Nilai Koefisien Silhouette")
                        ax_sil.set_ylabel("Label Klaster")
                        ax_sil.axvline(x=avg_silhouette_score, color="red", linestyle="--", label=f"Avg: {avg_silhouette_score:.2f}")
                        ax_sil.legend(loc='best')
                        ax_sil.set_yticks([])
                        ax_sil.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                        img_buffer = io.BytesIO()
                        fig_sil.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                        st.pyplot(fig_sil)
                        plt.close(fig_sil)
                        st.download_button(label="📥 Unduh Plot Silhouette (PNG)", data=img_buffer, file_name=f'silhouette_plot_{algo_name}_K{n_clusters}.png', mime='image/png')
                    else: st.info("Silhouette plot memerlukan setidaknya 2 klaster (non-noise).")
            with col2_sil:
                st.info("Plot ini membantu mengevaluasi seberapa baik setiap titik data agregat cocok dalam klasternya. Lebar bentuk menunjukkan skor (semakin mendekati 1 semakin baik). Garis merah adalah rata-rata skor.")

    except Exception as e:
        st.error(f"Terjadi error saat menampilkan hasil untuk {algo_name}: {e}")
        import traceback
        st.code(traceback.format_exc()) # Tampilkan traceback untuk debug

def display_trend_plots_for_region(result_package, selected_daerah):
    """
    Menampilkan set plot tren (line plot kecil) untuk satu algoritma 
    dan satu daerah yang dipilih.
    """
    try:
        # 1. Ekstrak data dari paket
        df_final_for_trend = result_package['df_final_labeled']
        results_for_trend = result_package['results_info']
        algo_name = result_package['algo_name'] # Ini akan jadi key prefix

        # 2. Buat master color map (spesifik untuk hasil ini)
        df_non_noise_trend = df_final_for_trend[df_final_for_trend['Cluster'] != -1]
        unique_clusters_trend = sorted(df_non_noise_trend['Cluster'].unique())
        if not unique_clusters_trend: # Jika hanya noise
            unique_clusters_trend = sorted(df_final_for_trend['Cluster'].unique())
            
        colors_sns_trend = sns.color_palette('viridis', n_colors=len(unique_clusters_trend))
        colors_hex_trend = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for (r,g,b) in colors_sns_trend]
        master_color_map_for_trend = {cluster: colors_hex_trend[i] for i, cluster in enumerate(unique_clusters_trend)}
        master_color_map_for_trend[-1] = 'grey'
        
        # 3. Judul Algoritma
        st.subheader(f"Tren untuk: {algo_name}")

        # 4. Ambil data daerah
        df_daerah = df_final_for_trend[df_final_for_trend['Kabupaten/Kota'] == selected_daerah].sort_values('Tahun')
        
        if not df_daerah.empty:
            daerah_cluster_label = df_daerah['Cluster_Label'].iloc[0]
            daerah_cluster_num = df_daerah['Cluster'].iloc[0] # Ambil angka klaster
            daerah_color = master_color_map_for_trend.get(daerah_cluster_num, 'grey') # Gunakan angka untuk ambil warna
            st.info(f"Daerah ini termasuk dalam: **{daerah_cluster_label}**")
            
            # 5. Loop fitur & plot
            fitur_rokok_plot = results_for_trend.get('selected_features_run', [])
            fitur_tambahan = []
            if 'IPM' in df_daerah.columns: fitur_tambahan.append('IPM')
            if 'Persentase_Miskin' in df_daerah.columns: fitur_tambahan.append('Persentase_Miskin')
            all_fitur_to_plot = fitur_rokok_plot + fitur_tambahan
            
            if all_fitur_to_plot:
                num_cols = 2 # 2 kolom plot kecil per algoritma
                cols = st.columns(num_cols)
                col_index = 0
                
                for fitur in all_fitur_to_plot:
                    if fitur in df_daerah.columns:
                        with cols[col_index]:
                            fig_line_daerah = px.line(df_daerah, x='Tahun', y=fitur, title=f"{fitur}", markers=True, height=400, color_discrete_sequence=[daerah_color], custom_data=['Cluster_Label', 'Provinsi', 'Kabupaten/Kota'])
                            
                            hover_label_fitur = "Nilai:"
                            if fitur == 'Persentase_Miskin': hover_label_fitur = "Kemiskinan (%):"
                            elif fitur == 'IPM': hover_label_fitur = "IPM:"
                            elif 'Rokok' in fitur or 'Tembakau' in fitur : hover_label_fitur = "Pengeluaran:"
                            
                            fig_line_daerah.update_traces(hovertemplate=f"<b>Klaster:</b> %{{customdata[0]}}<br><b>Provinsi:</b> %{{customdata[1]}}<br><b>Kab/Kota:</b> %{{customdata[2]}}<br><b>Tahun:</b> %{{x}}<br><b>{hover_label_fitur}</b> %{{y:,.2f}}<extra></extra>")
                            fig_line_daerah.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title=None, yaxis_title=None, dragmode='pan')
                            
                            # Buat key yang sangat unik
                            plot_key = f"trend_plot_{algo_name}_{fitur}_{selected_daerah.replace(' ', '_')}" 
                            st.plotly_chart(fig_line_daerah, use_container_width=True, key=plot_key)
                        col_index = (col_index + 1) % num_cols
                    else:
                         with cols[col_index]:
                             st.warning(f"Kolom '{fitur}' tidak ditemukan.")
                         col_index = (col_index + 1) % num_cols
            else:
                st.warning("Tidak ada fitur untuk diplot.")
        else:
            st.warning(f"Tidak ditemukan data tahunan untuk {selected_daerah} di hasil {algo_name}.")

    except Exception as e:
        st.error(f"Terjadi error saat menampilkan tren {algo_name}: {e}")
        import traceback
        st.code(traceback.format_exc()) # Tampilkan error untuk debug

# --- JUDUL UTAMA ---
st.title("Halaman Analisis Utama Klasterisasi 🔬")

# --- BAGIAN INPUT DATA ---
st.header("Sumber Data")
source_choice = st.radio(
    "Pilih sumber data untuk analisis:",
    ('Gunakan Dataset Bawaan Website', 'Unggah File Sendiri'),
    horizontal=True,
    key='data_source_radio',
    on_change=clear_analysis_results # Hapus hasil lama saat ganti
)

data_is_valid = False
temp_df_analysis = None 
source_name = "" # Untuk pesan error

if source_choice == 'Unggah File Sendiri':
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx'])
    if uploaded_file:
        temp_df_analysis = load_data(uploaded_file)
        source_name = f"File '{uploaded_file.name}'" # Nama untuk pesan error
else: 
    temp_df_analysis = load_data('dataset_rokok.csv')
    source_name = "Dataset Bawaan ('dataset_rokok.csv')" # Nama untuk pesan error

# --- PERBAIKAN: Lakukan Validasi Terpusat DI SINI ---
if temp_df_analysis is not None:
    # Cek jika load_data berhasil
    st.success(f"{source_name} berhasil dimuat. Memvalidasi...")
    
    validation_passed = True
    required_columns = ['Kabupaten/Kota', 'Tahun', 'ROKOK DAN TEMBAKAU', 'Rokok kretek filter', 'Rokok kretek tanpa filter', 'Rokok putih', 'Tembakau', 'Rokok dan tembakau Lainnya']
    min_rows_threshold = 10 
    
    # 1. Cek Kelengkapan Kolom
    missing_cols = [col for col in required_columns if col not in temp_df_analysis.columns]
    if missing_cols:
        st.error(f"❌ Validasi Gagal ({source_name}): Kolom berikut tidak ditemukan: **{', '.join(missing_cols)}**.")
        st.info(f"Pastikan file memiliki semua kolom ini: {', '.join(required_columns)}")
        validation_passed = False
    
    # 2. Cek Jumlah Baris Minimum (hanya jika kolom lengkap)
    if validation_passed and len(temp_df_analysis) < min_rows_threshold:
        st.error(f"❌ Validasi Gagal ({source_name}): Jumlah baris data terlalu sedikit ({len(temp_df_analysis)} baris). Minimal {min_rows_threshold}.")
        validation_passed = False
    
    # 3. Set Status Validasi Final
    if validation_passed:
        st.success(f"✅ Validasi Berhasil ({source_name})!")
        data_is_valid = True # Set flag valid
    else:
         temp_df_analysis = None # Kosongkan jika validasi gagal
         
elif (source_choice == 'Unggah File Sendiri' and uploaded_file):
    # Kasus jika load_data gagal (misal, file korup)
    st.error(f"Gagal memuat data dari file {uploaded_file.name}.")
elif (source_choice == 'Gunakan Dataset Bawaan Website'):
     # Kasus jika load_data bawaan gagal
     st.error("Gagal memuat dataset bawaan 'dataset_rokok.csv'.")

# --- SISA HALAMAN (HANYA TAMPIL JIKA DATA VALID) ---
if data_is_valid and temp_df_analysis is not None:
    # Simpan data valid ke session state
    st.session_state['df_analysis'] = temp_df_analysis
    st.write("Preview Data yang sedang aktif:")
    st.dataframe(st.session_state['df_analysis'].head())
    st.markdown("---") 

    # --- PENGATURAN ANALISIS (Input) ---
    st.header("Pengaturan Analisis")
    st.write("Pilihlah algoritma (maksimal 2) dan fitur sesuai keinginan anda")

    col1, col2, col3 = st.columns(3)
    with col1:
        algo_options = ['K-Means', 'K-Means++', 'OPTICS']
        algo_choices = st.multiselect(
            "Pilih Algoritma (Maks. 2):",
            options=algo_options,
            default=['K-Means'],
            max_selections=2,
            on_change=clear_analysis_results
        )
    with col2:
        all_features = ['ROKOK DAN TEMBAKAU', 'Rokok kretek filter', 'Rokok kretek tanpa filter', 'Rokok putih', 'Tembakau', 'Rokok dan tembakau Lainnya']
        default_features = ['Rokok kretek filter', 'Rokok kretek tanpa filter', 'Rokok putih', 'Tembakau', 'Rokok dan tembakau Lainnya']
        selected_features = st.multiselect("Pilih Fitur Analisis:", options=all_features, default=default_features, on_change=clear_analysis_results)
    with col3:
        start_year, end_year = st.slider("Pilih Rentang Tahun:", 2018, 2024, (2018, 2024), on_change=clear_analysis_results)
        selected_years = list(range(start_year, end_year + 1))

    st.markdown("---")

    # --- PENGATURAN PARAMETER ---
    st.header("Pengaturan Parameter & Klasterisasi")
    params = {} 
    
    kmeans_selected = 'K-Means' in algo_choices or 'K-Means++' in algo_choices
    if kmeans_selected:
        st.info("Tombol 'Cari Rekomendasi K' hanya berlaku untuk algoritma K-Means dan K-Means++.")
        if st.button("Cari rekomendasi nilai K"):
            df_rokok_eval = st.session_state['df_analysis'].copy()
            if selected_features:
                with st.spinner('Mengevaluasi K dari 2-10 pada data agregat...'):
                    df_filtered_eval_raw = df_rokok_eval[df_rokok_eval['Tahun'].isin(selected_years)].copy()
                    st.write("Mengagregasi data evaluasi (median)...")
                    features_to_agg_eval = selected_features
                    df_aggregated_eval = df_filtered_eval_raw.groupby('Kabupaten/Kota', as_index=False)[features_to_agg_eval].median()
                    df_aggregated_eval[features_to_agg_eval] = df_aggregated_eval[features_to_agg_eval].fillna(0)
                    scaler_eval = RobustScaler()
                    X_scaled_eval = scaler_eval.fit_transform(df_aggregated_eval[features_to_agg_eval])
                    k_range = range(2, 11)
                    scores_sklearn = []
                    scores_manual = []
                    for k in k_range:
                        km_sklearn = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
                        labels_sklearn = km_sklearn.fit_predict(X_scaled_eval)
                        scores_sklearn.append(silhouette_score(X_scaled_eval, labels_sklearn) if len(set(labels_sklearn)) > 1 else -1)
                        if KMeansPlusPlusManual:
                            km_manual = KMeansPlusPlusManual(n_clusters=k, random_state=42, n_init=10)
                            labels_manual = km_manual.fit_predict(X_scaled_eval)
                            scores_manual.append(silhouette_score(X_scaled_eval, labels_manual) if len(set(labels_manual)) > 1 else -1)
                    st.session_state['eval_graph_data'] = {'k_range': list(k_range), 'scores_sklearn': scores_sklearn, 'scores_manual': scores_manual}
                    st.success("Evaluasi K selesai.")
            else: st.warning("Pilih setidaknya satu fitur untuk evaluasi K.")

    # Tampilkan grafik evaluasi K (interaktif)
    if 'eval_graph_data' in st.session_state and kmeans_selected:
        eval_data = st.session_state['eval_graph_data']
        k_range = eval_data['k_range']
        scores_sklearn = eval_data['scores_sklearn']
        scores_manual = eval_data['scores_manual']
        plot_data = []
        for i, k in enumerate(k_range):
            if i < len(scores_sklearn): plot_data.append({'Jumlah Klaster (K)': k, 'Silhouette Score': scores_sklearn[i], 'Algoritma': 'K-Means'})
            if scores_manual and i < len(scores_manual): plot_data.append({'Jumlah Klaster (K)': k, 'Silhouette Score': scores_manual[i], 'Algoritma': 'K-Means++'})
        eval_df = pd.DataFrame(plot_data)
        if not eval_df.empty:
             fig_eval = px.line(eval_df, x='Jumlah Klaster (K)', y='Silhouette Score', color='Algoritma', markers=True, title="Perbandingan Silhouette Score (Interaktif)")
             fig_eval.update_layout(dragmode='pan')
             st.plotly_chart(fig_eval, use_container_width=True)
             st.info("Nilai K terbaik adalah yang memiliki Silhouette Score tertinggi.")
        
    # Tampilkan slider parameter
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        if kmeans_selected:
            st.subheader("Parameter K-Means / K-Means++")
            k_value_default = 3 # Default K
            if 'eval_graph_data' in st.session_state: # Coba ambil K terbaik dari eval
                 try:
                     all_scores = eval_data['scores_sklearn'] + eval_data['scores_manual']
                     best_score_index = np.argmax(all_scores) % len(eval_data['k_range'])
                     k_value_default = eval_data['k_range'][best_score_index]
                 except: pass # Gunakan default 3 jika gagal
            params['k'] = st.slider("Pilih Jumlah Klaster (K):", 2, 10, k_value_default, help="Lihat rekomendasi dari tombol 'Cari K' di atas.", on_change=clear_analysis_results)
    with param_col2:
        if 'OPTICS' in algo_choices:
            st.subheader("Parameter OPTICS")
            default_min_samples = max(2, len(selected_features) * 2) if selected_features else 5
            help_text_optics = f"Rekomendasi default (2 x Jml Fitur): {default_min_samples}"
            params['min_samples'] = st.slider("Pilih Jumlah Sampel Minimum (min_samples):", min_value=2, max_value=50, value=default_min_samples, help=help_text_optics, on_change=clear_analysis_results)
            st.caption(f"Rekomendasi `min_samples` untuk {len(selected_features)} fitur adalah **{default_min_samples}**.")


    # --- PROSES KLASTERISASI (Tombol) ---
    if st.button("🚀 Proses Klasterisasi", type="primary", key='process_button_multi'):
        
        # Validasi
        if not algo_choices:
            st.warning("⚠️ Harap pilih setidaknya satu algoritma.")
        elif not selected_features:
            st.warning("⚠️ Harap pilih setidaknya satu fitur.")
        elif 'K-Means' in algo_choices and 'k' not in params:
             st.warning("⚠️ Harap atur parameter 'Jumlah Klaster (K)'.")
        elif 'K-Means++' in algo_choices and 'k' not in params:
             st.warning("⚠️ Harap atur parameter 'Jumlah Klaster (K)'.")
        elif 'OPTICS' in algo_choices and 'min_samples' not in params:
             st.warning("⚠️ Harap atur parameter 'Jumlah Sampel Minimum'.")
        else:
            # Validasi lolos, jalankan proses
            df_to_process = st.session_state['df_analysis']
            df_tambahan_orig = load_data('dataset_tambahan.csv') # Muat data tambahan sekali
            
            with st.spinner(f'Melakukan analisis untuk: {", ".join(algo_choices)}...'):
                
                # 1. Agregasi Data (Satu Kali)
                df_filtered = df_to_process[df_to_process['Tahun'].isin(selected_years)].copy()
                st.write("Mengagregasi data (median)...")
                features_to_agg = selected_features
                agg_dict = {feat: 'median' for feat in features_to_agg}
                if 'Provinsi' in df_filtered.columns: agg_dict['Provinsi'] = 'first' # Ambil Provinsi
                df_aggregated = df_filtered.groupby('Kabupaten/Kota', as_index=False).agg(agg_dict)
                
                results_list = [] # List untuk menyimpan hasil

                # 2. Loop per Algoritma
                for algo_name in algo_choices:
                    st.write(f"--- Menjalankan {algo_name} ---")
                    start_time = time.time()
                    
                    # 3. Jalankan Klasterisasi
                    clustering_results = run_clustering(df_aggregated.copy(), selected_features, algo_name, params)
                    runtime = time.time() - start_time
                    
                    if clustering_results:
                        clustering_results['runtime'] = runtime
                        clustering_results['start_year'] = start_year
                        clustering_results['end_year'] = end_year
                        
                        df_result_agg = clustering_results['df_result'] # Hasil agregat + klaster
                        
                        # 4. Buat Label Teks
                        labels_numeric = df_result_agg['Cluster'].unique()
                        label_map = {cluster: f"Klaster {cluster}" if cluster != -1 else "Noise" for cluster in sorted(labels_numeric)}
                        
                        # Tambahkan label teks ke df_result_agg (untuk plot PCA/Strip)
                        df_result_agg['Cluster_Label'] = df_result_agg['Cluster'].map(label_map)
                        clustering_results['df_result'] = df_result_agg # Simpan kembali
                        
                        # 5. Buat df_final (Data Asli + Label)
                        cluster_map = df_result_agg.set_index('Kabupaten/Kota')['Cluster'].to_dict()
                        df_final_labeled = df_filtered.copy()
                        df_final_labeled['Cluster'] = df_final_labeled['Kabupaten/Kota'].map(cluster_map)
                        df_final_labeled['Cluster_Label'] = df_final_labeled['Cluster'].map(label_map)
                        
                        # 6. Gabungkan dengan Data Tambahan
                        if df_tambahan_orig is not None:
                            df_tambahan_filtered = df_tambahan_orig[df_tambahan_orig['Tahun'].isin(selected_years)].copy()
                            df_final_labeled = pd.merge(df_final_labeled, df_tambahan_filtered, on=['Kabupaten/Kota', 'Tahun'], how='left')
                        
                        # 7. Tambahkan Kunci Peta
                        df_final_labeled['key_df'] = df_final_labeled['Kabupaten/Kota'].str.upper()
                        df_final_labeled['id_key'] = df_final_labeled['Kabupaten/Kota'].str.upper()
                        
                        # 8. Buat Paket Hasil
                        result_package = {
                            'algo_name': algo_name,
                            'results_info': clustering_results, # Berisi df_result_agg, metrics, model, dll.
                            'df_final_labeled': df_final_labeled # Berisi data asli + label + tambahan
                        }
                        results_list.append(result_package)
                    
                    else:
                        st.error(f"Gagal menjalankan klasterisasi untuk {algo_name}.")
                
                # 9. Simpan list hasil ke session state
                if results_list:
                    st.session_state['analysis_results_list'] = results_list
                    st.success("Analisis selesai!")
                else:
                    st.error("Semua proses klasterisasi gagal.")

# --- BAGIAN HASIL ANALISIS (BARU) ---
# Tampilkan jika list hasil ada di session state
if 'analysis_results_list' in st.session_state:
    all_results = st.session_state['analysis_results_list']
    
    if len(all_results) == 1:
        # Jika 1 hasil, tampilkan penuh
        st.header(f"Hasil Analisis: {all_results[0]['algo_name']}")
        st.markdown("---")
        display_analysis_results(all_results[0]) # Panggil fungsi display
    
    elif len(all_results) == 2:
        # Jika 2 hasil, buat 2 kolom
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.header(f"Hasil Analisis: {all_results[0]['algo_name']}")
            st.markdown("---")
            display_analysis_results(all_results[0]) # Panggil fungsi display
        with col_res2:
            st.header(f"Hasil Analisis: {all_results[1]['algo_name']}")
            st.markdown("---")
            display_analysis_results(all_results[1]) # Panggil fungsi display

# --- Tampilkan jika data tidak valid ---
elif source_choice == 'Unggah File Sendiri' and uploaded_file and temp_df_analysis is None and not data_is_valid:
    st.warning("Perbaiki masalah pada file yang diunggah sebelum melanjutkan.")
elif source_choice == 'Gunakan Dataset Bawaan Website' and not data_is_valid:
     st.error("Gagal memuat dataset bawaan 'dataset_rokok.csv'. Periksa apakah file ada.")

# --- FITUR BARU: Analisis Tren per Daerah ---
st.markdown("---") 
st.header("🔍 Analisis Tren Tahunan per Daerah")
st.write("Pilih satu Kabupaten/Kota untuk melihat tren tahunan berdasarkan hasil klasterisasi.")

# Cek apakah hasil analisis sudah ada di session state
if 'analysis_results_list' in st.session_state and st.session_state['analysis_results_list']:
    all_results = st.session_state['analysis_results_list']
    
    # Ambil df_final *pertama* HANYA untuk membuat daftar selectbox
    # Daftar daerah akan sama untuk semua hasil
    df_final_for_list = all_results[0]['df_final_labeled']
    list_daerah = sorted([daerah for daerah in df_final_for_list['Kabupaten/Kota'].unique() if isinstance(daerah, str)])

    if not list_daerah:
        st.warning("Tidak ada nama daerah valid ditemukan dalam data hasil.")
    else:
        # 1. Buat Selectbox di luar kolom
        selected_daerah = st.selectbox(
            "Pilih Kabupaten/Kota:",
            options=['-- Pilih Daerah --'] + list_daerah,
            index=0,
            key="trend_daerah_selector" # Key unik untuk selectbox
        )

        # 2. Hanya tampilkan plot jika daerah sudah dipilih
        if selected_daerah != '-- Pilih Daerah --':
            
            # 3. Tampilkan hasil berdasarkan jumlah algoritma
            if len(all_results) == 2:
                # Jika ada 2 hasil, buat 2 kolom
                col_trend_1, col_trend_2 = st.columns(2)
                with col_trend_1:
                    # Panggil fungsi helper untuk hasil pertama
                    display_trend_plots_for_region(all_results[0], selected_daerah)
                with col_trend_2:
                    # Panggil fungsi helper untuk hasil kedua
                    display_trend_plots_for_region(all_results[1], selected_daerah)
                    
            elif len(all_results) == 1:
                # Jika hanya 1 hasil, tampilkan biasa (tidak perlu kolom)
                display_trend_plots_for_region(all_results[0], selected_daerah)
else:
    st.info("Jalankan proses klasterisasi terlebih dahulu untuk dapat melihat tren per daerah.")
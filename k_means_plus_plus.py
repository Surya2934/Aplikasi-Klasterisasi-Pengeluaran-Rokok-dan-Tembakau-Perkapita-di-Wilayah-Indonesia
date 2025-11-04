import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

class KMeansPlusPlusManual:

    def __init__(self, n_clusters, max_iter=300, random_state=None, n_init=10):
        if n_clusters <= 0:
            raise ValueError("Jumlah klaster harus positif.")
        if max_iter <= 0:
             raise ValueError("Iterasi maksimum harus positif.")
        if n_init <= 0:
             raise ValueError("Jumlah inisialisasi harus positif.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)
        self.n_init = n_init
        self.centroids = None
        self.labels_ = None
        self.inertia_ = float('inf')
        self.n_iter_ = 0

    #Fungsi Euclidean Distances
    def _euclidean_distance(self, point1, point2):
        return np.sum((point1 - point2)**2)

    def _init_centroids(self, X):
        #Menginisialisasi centroid menggunakan metode K-Means++
        n_samples, n_features = X.shape
        centroids = []

        #1. Memilih centroid pertama secara acak dari titik data
        first_centroid_idx = self.random_state.randint(n_samples)
        centroids.append(X[first_centroid_idx])

        #2. Pilih sisa centroid (n_clusters - 1)
        for _ in range(1, self.n_clusters):
            # Hitung kuadrat jarak dari setiap titik ke centroid terdekat *yang sudah ada*
            distances = np.array([[self._euclidean_distance(dp, c) for c in centroids] for dp in X])

            # min_dist_sq berbentuk (n_samples,) - berisi kuadrat jarak terpendek untuk setiap titik
            min_dist_sq = np.min(distances, axis=1)

            # Hitung probabilitas yang proporsional dengan D(x)^2
            total_dist_sq = np.sum(min_dist_sq)
            if total_dist_sq == 0: # Hindari pembagian dengan nol jika semua titik sama
                 probabilities = np.ones(n_samples) / n_samples
            else:
                 probabilities = min_dist_sq / total_dist_sq

            # Pilih centroid berikutnya berdasarkan probabilitas
            probabilities /= probabilities.sum()
            next_centroid_index = self.random_state.choice(n_samples, p=probabilities)
            centroids.append(X[next_centroid_index])

        return np.array(centroids)

    def _assign_clusters(self, X, centroids):
        #Menugaskan setiap titik data ke centroid terdekat-nya
        #Ini mengembalikan array di mana labels_[i] adalah indeks centroid terdekat ke X[i]
        labels, _ = pairwise_distances_argmin_min(X, centroids, metric='euclidean')
        return labels

    def _update_centroids(self, X, labels, old_centroids):
        #Memperbarui centroid berdasarkan rata-rata titik di setiap klaster
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        for i in range(self.n_clusters):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0: # Cek jika klaster tidak kosong
                new_centroids[i] = np.mean(points_in_cluster, axis=0)
            else:
                #Tangani klaster kosong: pertahankan posisi centroid lama
                new_centroids[i] = old_centroids[i]
        return new_centroids

    def _calculate_inertia(self, X, centroids, labels):
        #Menghitung within-cluster sum of squares (inersia)
        inertia = 0
        for i in range(self.n_clusters):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                centroid = centroids[i]
                # Jumlahkan kuadrat jarak dari titik ke centroid yang ditugaskan
                inertia += np.sum((points_in_cluster - centroid)**2)
        return inertia

    def fit(self, X):
        # Validasi input (dasar)
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if len(X.shape) != 2:
             raise ValueError("Input X harus 2 dimensi.")
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"Jumlah sampel ({X.shape[0]}) harus >= jumlah klaster ({self.n_clusters}).")

        # Jalankan algoritma n_init kali dan simpan hasil terbaik
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        for run in range(self.n_init):
            #1. Inisialisasi menggunakan K-Means++
            current_centroids = self._init_centroids(X)
            current_labels = np.zeros(X.shape[0], dtype=int) # Gunakan int untuk label
            run_n_iter = 0

            #2. Iterasi K-Means seperti biasa
            for i in range(self.max_iter):
                old_centroids = np.copy(current_centroids)

                #Tugaskan titik ke klaster
                current_labels = self._assign_clusters(X, current_centroids)

                #Perbarui centroid
                current_centroids = self._update_centroids(X, current_labels, old_centroids)

                run_n_iter = i + 1

                #Cek konvergensi
                if np.allclose(current_centroids, old_centroids, atol=1e-4):
                    break
            else: #Dijalankan jika loop selesai tanpa break
                # Pastikan label diperbarui berdasarkan centroid terakhir jika max_iter tercapai
                 current_labels = self._assign_clusters(X, current_centroids)


            # 3.Hitung Inersia untuk proses (run) ini
            current_inertia = self._calculate_inertia(X, current_centroids, current_labels)

            # 4.Simpan catatan proses terbaik
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_centroids = current_centroids
                best_labels = current_labels
                best_n_iter = run_n_iter

        # 5.Tetapkan atribut final berdasarkan proses (run) terbaik
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.n_iter_ = best_n_iter
        self.inertia_ = best_inertia

        return self

    def fit_predict(self, X):
        #Hitung pusat klaster dan prediksi indeks klaster untuk setiap sampel.
        self.fit(X)
        return self.labels_
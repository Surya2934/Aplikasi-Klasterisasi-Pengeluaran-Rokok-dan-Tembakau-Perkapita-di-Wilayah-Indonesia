import numpy as np

class KMeansPlusPlusManual:
    """
    Implementasi manual dari algoritma K-Means++ dari nol.
    """
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None # Akan berisi centroid akhir (numerik)
        self.labels_ = None
        self.n_iter_ = 0
        self.initial_centroid_indices_ = [] # Menyimpan indeks data asli yang menjadi centroid awal

    def _initialize_centroids(self, X):
        """Inisialisasi centroid menggunakan metode K-Means++ dan simpan indeksnya."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.initial_centroid_indices_ = []

        # 1. Pilih centroid pertama secara acak
        first_idx = np.random.choice(X.shape[0], 1)[0]
        centroids = [X[first_idx]]
        self.initial_centroid_indices_.append(first_idx)

        # 2. Pilih sisa centroid
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            
            next_centroid_idx = 0
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    next_centroid_idx = j
                    break
            
            centroids.append(X[next_centroid_idx])
            self.initial_centroid_indices_.append(next_centroid_idx)
            
        return np.array(centroids)

    def fit_predict(self, X):
        """Menjalankan algoritma K-Means++ dan mengembalikan label klaster."""
        # Inisialisasi centroid dan simpan indeksnya
        initial_centroids = self._initialize_centroids(X)
        self.centroids = np.copy(initial_centroids)
        
        self.n_iter_ = 0
        for i in range(self.max_iter):
            self.n_iter_ = i + 1
            labels = np.array([np.argmin([np.linalg.norm(x-c) for c in self.centroids]) for x in X])
            
            new_centroids = np.array([X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else self.centroids[k] for k in range(self.n_clusters)])
            
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
            
        self.labels_ = labels
        return self.labels_


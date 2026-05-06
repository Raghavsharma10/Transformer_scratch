def fit(self, X):
        """ Apply KMeans Clustering
              X: dataset with feature vectors
        """
        self.centers_, self.labels_, self.sse_arr_, self.n_iter_ = \
              _kmedoids(X, self.n_clusters, self.distance, self.max_iter, self.n_trials, self.tol, self.rng)
def fit(self, X):
        """ Apply KMeans Clustering
              X: dataset with feature vectors
        """
        self.centers_, self.labels_, self.sse_arr_, self.n_iter_ = \
              _kmeans(X, self.n_clusters, self.max_iter, self.n_trials, self.tol)
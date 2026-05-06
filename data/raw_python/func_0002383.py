def _calculate_gap(self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, n_clusters: int) -> Tuple[float, int]:
        """
        Calculate the gap value of the given data, n_refs, and number of clusters.
        Return the resutling gap value and n_clusters
        """
        # Holder for reference dispersion results
        ref_dispersions = np.zeros(n_refs)  # type: np.ndarray

        # For n_references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(n_refs):

            # Create new random reference set
            random_data = np.random.random_sample(size=X.shape)  # type: np.ndarray

            # Fit to it, getting the centroids and labels, and add to accumulated reference dispersions array.
            centroids, labels = kmeans2(data=random_data,
                                        k=n_clusters,
                                        iter=10,
                                        minit='points')  # type: Tuple[np.ndarray, np.ndarray]
            dispersion = self._calculate_dispersion(X=random_data, labels=labels, centroids=centroids)  # type: float
            ref_dispersions[i] = dispersion

        # Fit cluster to original data and create dispersion calc.
        centroids, labels = kmeans2(data=X, k=n_clusters, iter=10, minit='points')
        dispersion = self._calculate_dispersion(X=X, labels=labels, centroids=centroids)

        # Calculate gap statistic
        gap_value = np.mean(np.log(ref_dispersions)) - np.log(dispersion)

        return gap_value, int(n_clusters)
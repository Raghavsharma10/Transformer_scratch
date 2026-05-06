def _process_non_parallel(self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, cluster_array: np.ndarray):
        """
        Process calling of .calculate_gap() method using no parallel backend; simple for loop generator
        """
        for gap_value, n_clusters in [self._calculate_gap(X, n_refs, n_clusters)
                                      for n_clusters in cluster_array]:
            yield (gap_value, n_clusters)
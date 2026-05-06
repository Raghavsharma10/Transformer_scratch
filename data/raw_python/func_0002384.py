def _process_with_rust(self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, cluster_array: np.ndarray):
        """
        Process gap stat using pure rust
        """
        from gap_statistic.rust import gapstat
        for label, gap_value in gapstat.optimal_k(X, list(cluster_array)):
            yield (gap_value, label)
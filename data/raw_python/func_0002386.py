def _process_with_multiprocessing(self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, cluster_array: np.ndarray):
        """
        Process calling of .calculate_gap() method using the multiprocessing library
        """
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:

            jobs = [executor.submit(self._calculate_gap, X, n_refs, n_clusters)
                    for n_clusters in cluster_array
                    ]

            for future in as_completed(jobs):
                gap_value, k = future.result()
                yield (gap_value, k)
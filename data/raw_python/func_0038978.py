def get_data_matrix_in_order(self, subset_ids):
        """
        Returns a numpy array of features, rows in the same order as subset_ids

        Parameters
        ----------
        subset_ids : list
            List od sample IDs to extracted from the dataset.

        Returns
        -------
        matrix : ndarray
            Matrix of features, for each id in subset_ids, in order.
        """

        if len(subset_ids) < 1:
            warnings.warn('subset must have atleast one ID - returning empty matrix!')
            return np.empty((0, 0))

        if isinstance(subset_ids, set):
            raise TypeError('Input set is not ordered, hence can not guarantee order! '
                            'Must provide a list or tuple.')

        if isinstance(subset_ids, str):
            subset_ids = [subset_ids, ]

        num_existing_keys = sum([1 for key in subset_ids if key in self.__data])
        if num_existing_keys < len(subset_ids):
            raise ValueError('One or more IDs from  subset do not exist in the dataset!')

        matrix = np.full((num_existing_keys, self.num_features), np.nan)
        for idx, sid in enumerate(subset_ids):
            matrix[idx, :] = self.__data[sid]

        return matrix
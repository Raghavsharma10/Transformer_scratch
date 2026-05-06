def data_and_labels(self):
        """
        Dataset features and labels in a matrix form for learning.

        Also returns sample_ids in the same order.

        Returns
        -------
        data_matrix : ndarray
            2D array of shape [num_samples, num_features]
            with features corresponding row-wise to sample_ids
        labels : ndarray
            Array of numeric labels for each sample corresponding row-wise to sample_ids
        sample_ids : list
            List of sample ids

        """

        sample_ids = np.array(self.keys)
        label_dict = self.labels
        matrix = np.full([self.num_samples, self.num_features], np.nan)
        labels = np.full([self.num_samples, 1], np.nan)
        for ix, sample in enumerate(sample_ids):
            matrix[ix, :] = self.__data[sample]
            labels[ix] = label_dict[sample]

        return matrix, np.ravel(labels), sample_ids
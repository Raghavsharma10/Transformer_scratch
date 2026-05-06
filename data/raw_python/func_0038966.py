def check_features(self, features):
        """
        Method to ensure data to be added is not empty and vectorized.

        Parameters
        ----------
        features : iterable
            Any data that can be converted to a numpy array.

        Returns
        -------
        features : numpy array
            Flattened non-empty numpy array.

        Raises
        ------
        ValueError
            If input data is empty.
        """

        if not isinstance(features, np.ndarray):
            features = np.asarray(features)

        if features.size <= 0:
            raise ValueError('provided features are empty.')

        if features.ndim > 1:
            features = np.ravel(features)

        return features
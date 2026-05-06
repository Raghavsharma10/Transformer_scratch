def return_features_numpy(self, names='all'):
        """
        Returns a 2d numpy array of extracted features

        Parameters
        ----------
        names : list of strings, a list of feature names which are to be retrieved from the database, if equal to 'all',
        all features will be returned, default value: 'all'

        Returns
        -------
        A numpy array of features, each row corresponds to a single datapoint. If a single feature is a 1d numpy array,
        then it will be unrolled into the resulting array. Higher-dimensional numpy arrays are not supported.
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return return_features_numpy_base(self.dbpath, self._set_object, self.points_amt, names)
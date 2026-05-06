def return_features(self, names='all'):
        """
        Returns a list of extracted features from the database

        Parameters
        ----------
        names : list of strings, a list of feature names which are to be retrieved from the database, if equal
        to 'all', the all features will be returned, default value: 'all'

        Returns
        -------
        A list of lists, each 'inside list' corresponds to a single data point, each element of the 'inside list' is a
        feature (can be of any type)
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return return_features_base(self.dbpath, self._set_object, names)
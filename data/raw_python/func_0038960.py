def data(self, values, feature_names=None):
        """
        Populates this dataset with the provided data.
        Usage of this method is discourage (unless you know what you are doing).

        Parameters
        ----------
        values : dict
            dict of features keyed in by sample ids.

        feature_names : list of str
            New feature names for the new features, if available.

        Raises
        ------
        ValueError
            If number of samples does not match the size of existing set, or
            If atleast one sample is not provided.

        """
        if isinstance(values, dict):
            if self.__labels is not None and len(self.__labels) != len(values):
                raise ValueError(
                    'number of samples do not match the previously assigned labels')
            elif len(values) < 1:
                raise ValueError('There must be at least 1 sample in the dataset!')
            else:
                self.__data = values
                # update dimensionality
                # assuming all keys in dict have same len arrays
                self.__num_features = len(values[self.keys[0]])

            if feature_names is None:
                self.__feature_names = self.__str_names(self.num_features)
            else:
                self.feature_names = feature_names
        else:
            raise ValueError('data input must be a dictionary!')
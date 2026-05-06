def add_sample(self, sample_id, features, label,
                   class_id=None,
                   overwrite=False,
                   feature_names=None):
        """Adds a new sample to the dataset with its features, label and class ID.

        This is the preferred way to construct the dataset.

        Parameters
        ----------

        sample_id : str, int
            The identifier that uniquely identifies this sample.
        features : list, ndarray
            The features for this sample
        label : int, str
            The label for this sample
        class_id : int, str
            The class for this sample.
            If not provided, label converted to a string becomes its ID.
        overwrite : bool
            If True, allows the overwite of features for an existing subject ID.
            Default : False.
        feature_names : list
            The names for each feature. Assumed to be in the same order as `features`

        Raises
        ------
        ValueError
            If `sample_id` is already in the MLDataset (and overwrite=False), or
            If dimensionality of the current sample does not match the current, or
            If `feature_names` do not match existing names
        TypeError
            If sample to be added is of different data type compared to existing samples.

        """

        if sample_id in self.__data and not overwrite:
            raise ValueError('{} already exists in this dataset!'.format(sample_id))

        # ensuring there is always a class name, even when not provided by the user.
        # this is needed, in order for __str__ method to work.
        # TODO consider enforcing label to be numeric and class_id to be string
        #  so portability with other packages is more uniform e.g. for use in scikit-learn
        if class_id is None:
            class_id = str(label)

        features = self.check_features(features)
        if self.num_samples <= 0:
            self.__data[sample_id] = features
            self.__labels[sample_id] = label
            self.__classes[sample_id] = class_id
            self.__dtype = type(features)
            self.__num_features = features.size if isinstance(features,
                                                              np.ndarray) else len(
                features)
            if feature_names is None:
                self.__feature_names = self.__str_names(self.num_features)
        else:
            if self.__num_features != features.size:
                raise ValueError('dimensionality of this sample ({}) '
                                 'does not match existing samples ({})'
                                 ''.format(features.size, self.__num_features))
            if not isinstance(features, self.__dtype):
                raise TypeError("Mismatched dtype. Provide {}".format(self.__dtype))

            self.__data[sample_id] = features
            self.__labels[sample_id] = label
            self.__classes[sample_id] = class_id
            if feature_names is not None:
                # if it was never set, allow it
                # class gets here when adding the first sample,
                #   after dataset was initialized with empty constructor
                if self.__feature_names is None:
                    self.__feature_names = np.array(feature_names)
                else:  # if set already, ensure a match
                    if not np.array_equal(self.feature_names, np.array(feature_names)):
                        raise ValueError(
                            "supplied feature names do not match the existing names!")
def get_feature_subset(self, subset_idx):
        """
        Returns the subset of features indexed numerically.

        Parameters
        ----------
        subset_idx : list, ndarray
            List of indices to features to be returned

        Returns
        -------
        MLDataset : MLDataset
            with subset of features requested.

        Raises
        ------
        UnboundLocalError
            If input indices are out of bounds for the dataset.

        """

        subset_idx = np.asarray(subset_idx)
        if not (max(subset_idx) < self.__num_features) and (min(subset_idx) >= 0):
            raise UnboundLocalError('indices out of range for the dataset. '
                                    'Max index: {} Min index : 0'.format(
                self.__num_features))

        sub_data = {sample: features[subset_idx] for sample, features in
                    self.__data.items()}
        new_descr = 'Subset features derived from: \n ' + self.__description
        subdataset = MLDataset(data=sub_data,
                               labels=self.__labels, classes=self.__classes,
                               description=new_descr,
                               feature_names=self.__feature_names[subset_idx])

        return subdataset
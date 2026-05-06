def get_subset(self, subset_ids):
        """
        Returns a smaller dataset identified by their keys/sample IDs.

        Parameters
        ----------
        subset_ids : list
            List od sample IDs to extracted from the dataset.

        Returns
        -------
        sub-dataset : MLDataset
            sub-dataset containing only requested sample IDs.

        """

        num_existing_keys = sum([1 for key in subset_ids if key in self.__data])
        if subset_ids is not None and num_existing_keys > 0:
            # ensure items are added to data, labels etc in the same order of sample IDs
            # TODO come up with a way to do this even when not using OrderedDict()
            # putting the access of data, labels and classes in the same loop  would
            # ensure there is correspondence across the three attributes of the class
            data = self.__get_subset_from_dict(self.__data, subset_ids)
            labels = self.__get_subset_from_dict(self.__labels, subset_ids)
            if self.__classes is not None:
                classes = self.__get_subset_from_dict(self.__classes, subset_ids)
            else:
                classes = None
            subdataset = MLDataset(data=data, labels=labels, classes=classes)
            # Appending the history
            subdataset.description += '\n Subset derived from: ' + self.description
            subdataset.feature_names = self.__feature_names
            subdataset.__dtype = self.dtype
            return subdataset
        else:
            warnings.warn('subset of IDs requested do not exist in the dataset!')
            return MLDataset()
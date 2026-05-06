def set_default_by_index(self, index):
        """ Set the default dataset by its index.

        After changing the default dataset, all calls without explicitly specifying the
        dataset by index or alias will be redirected to this dataset.

        Args:
            index (int): The index of the dataset that should be made the default.

        Raises:
            DataInvalidIndex: If the index does not represent a valid dataset.
        """
        if index >= len(self._datasets):
            raise DataInvalidIndex('A dataset with index {} does not exist'.format(index))

        self._default_index = index
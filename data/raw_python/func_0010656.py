def get_by_index(self, index):
        """ Return a dataset by its index.

        Args:
            index (int): The index of the dataset that should be returned.

        Raises:
            DataInvalidIndex: If the index does not represent a valid dataset.
        """
        if index >= len(self._datasets):
            raise DataInvalidIndex('A dataset with index {} does not exist'.format(index))

        return self._datasets[index]
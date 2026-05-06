def add_alias(self, alias, index):
        """ Add an alias pointing to the specified index.

        Args:
            alias (str): The alias that should point to the given index.
            index (int): The index of the dataset for which an alias should be added.

        Raises:
            DataInvalidIndex: If the index does not represent a valid dataset.
        """
        if index >= len(self._datasets):
            raise DataInvalidIndex('A dataset with index {} does not exist'.format(index))
        self._aliases[alias] = index
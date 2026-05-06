def get_by_alias(self, alias):
        """ Return a dataset by its alias.

        Args:
            alias (str): The alias of the dataset that should be returned.

        Raises:
            DataInvalidAlias: If the alias does not represent a valid dataset.
        """
        if alias not in self._aliases:
            raise DataInvalidAlias('A dataset with alias {} does not exist'.format(alias))

        return self.get_by_index(self._aliases[alias])
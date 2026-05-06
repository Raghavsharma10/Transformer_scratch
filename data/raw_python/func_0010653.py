def set_default_by_alias(self, alias):
        """ Set the default dataset by its alias.

        After changing the default dataset, all calls without explicitly specifying the
        dataset by index or alias will be redirected to this dataset.

        Args:
            alias (str): The alias of the dataset that should be made the default.

        Raises:
            DataInvalidAlias: If the alias does not represent a valid dataset.
        """
        if alias not in self._aliases:
            raise DataInvalidAlias('A dataset with alias {} does not exist'.format(alias))

        self._default_index = self._aliases[alias]
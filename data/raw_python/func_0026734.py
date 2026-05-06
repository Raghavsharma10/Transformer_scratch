def add_alias(self, alias, source, clean=True):
        """Add an alias, optionally 'cleaning' the alias string.

        Calls the parent `catalog` method `clean_entry_name` - to apply the
        same name-cleaning as is applied to entry names themselves.

        Returns
        -------
        alias : str
            The stored version of the alias (cleaned or not).

        """
        if clean:
            alias = self.catalog.clean_entry_name(alias)
        self.add_quantity(self._KEYS.ALIAS, alias, source)
        return alias
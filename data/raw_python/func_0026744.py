def get_aliases(self, includename=True):
        """Retrieve the aliases of this object as a list of strings.

        Arguments
        ---------
        includename : bool
            Include the 'name' parameter in the list of aliases.
        """
        # empty list if doesnt exist
        alias_quanta = self.get(self._KEYS.ALIAS, [])
        aliases = [aq[QUANTITY.VALUE] for aq in alias_quanta]
        if includename and self[self._KEYS.NAME] not in aliases:
            aliases = [self[self._KEYS.NAME]] + aliases
        return aliases
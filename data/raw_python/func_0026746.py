def get_source_by_alias(self, alias):
        """Given an alias, find the corresponding source in this entry.

        If the given alias doesn't exist (e.g. there are no sources), then a
        `ValueError` is raised.

        Arguments
        ---------
        alias : str
            The str-integer (e.g. '8') of the target source.

        Returns
        -------
        source : `astrocats.catalog.source.Source` object
            The source object corresponding to the passed alias.

        """
        for source in self.get(self._KEYS.SOURCES, []):
            if source[self._KEYS.ALIAS] == alias:
                return source
        raise ValueError("Source '{}': alias '{}' not found!".format(self[
            self._KEYS.NAME], alias))
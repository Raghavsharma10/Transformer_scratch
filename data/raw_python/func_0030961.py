def to_list(self):
        """Converts the GeneSet object to a flat list of strings.

        Note: see also :meth:`from_list`.

        Parameters
        ----------

        Returns
        -------
        list of str
            The data from the GeneSet object as a flat list.
        """
        src = self._source or ''
        coll = self._collection or ''
        desc = self._description or ''

        l = [self._id, src, coll, self._name,
             ','.join(sorted(self._genes)), desc]
        return l
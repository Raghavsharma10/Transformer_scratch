def _lowest_rowids(self, table, limit):
        """
        Gets the lowest available row ids for table insertion. Keeps things tidy!

        Parameters
        ----------
        table: str
            The name of the table being modified
        limit: int
            The number of row ids needed

        Returns
            -------
        available: sequence
            An array of all available row ids

        """
        try:
            t = self.query("SELECT id FROM {}".format(table), unpack=True, fmt='table')
            ids = t['id']
            all_ids = np.array(range(1, max(ids)))
        except TypeError:
            ids = None
            all_ids = np.array(range(1, limit+1))

        available = all_ids[np.in1d(all_ids, ids, assume_unique=True, invert=True)][:limit]

        # If there aren't enough empty row ids, start using the new ones
        if len(available) < limit:
            diff = limit - len(available)
            available = np.concatenate((available, np.array(range(max(ids) + 1, max(ids) + 1 + diff))))

        return available
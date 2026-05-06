def findUnique(self, tableClass, comparison=None, default=_noItem):
        """
        Find an Item in the database which should be unique.  If it is found,
        return it.  If it is not found, return 'default' if it was passed,
        otherwise raise L{errors.ItemNotFound}.  If more than one item is
        found, raise L{errors.DuplicateUniqueItem}.

        @param comparison: implementor of L{iaxiom.IComparison}.

        @param default: value to use if the item is not found.
        """
        results = list(self.query(tableClass, comparison, limit=2))
        lr = len(results)

        if lr == 0:
            if default is _noItem:
                raise errors.ItemNotFound(comparison)
            else:
                return default
        elif lr == 2:
            raise errors.DuplicateUniqueItem(comparison, results)
        elif lr == 1:
            return results[0]
        else:
            raise AssertionError("limit=2 database query returned 3+ results: ",
                                 comparison, results)
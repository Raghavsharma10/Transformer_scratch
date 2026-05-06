def _get_pseudo_key(self, row):
        """
        Returns the pseudo key in a row.

        :param dict row: The row.

        :rtype: tuple
        """
        ret = list()
        for key in self._pseudo_key:
            ret.append(row[key])

        return tuple(ret)
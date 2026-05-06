def hierarchy(self, alias=None):
        """
        return the URL for the bank hierarchy itself
        :param alias:
        :return:
        """
        if alias:
            return self._root + self._safe_alias(alias)
        else:
            return self._root
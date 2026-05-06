def parents(self, alias, bank_id):
        """
        URL for getting or setting parent relationships for the specified bank
        :param alias:
        :param bank_id:
        :return:
        """
        return self._root + self._safe_alias(alias) + '/parent/ids/' + bank_id
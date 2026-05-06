def children(self, alias, bank_id):
        """
        URL for getting or setting child relationships for the specified bank
        :param alias:
        :param bank_id:
        :return:
        """
        return self._root + self._safe_alias(alias) + '/child/ids/' + str(bank_id)
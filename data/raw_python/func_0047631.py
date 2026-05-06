def nodes(self, alias, depth=10, bank_id=None):
        """
        URL for getting bulk nodes in hierarchy
        :param alias:
        :param depth:
        :return:
        """
        if bank_id:
            return self._root + self._safe_alias(alias) + '/child/nodes/' + bank_id + '?descendentlevels=' + str(depth)
        else:
            return self._root + self._safe_alias(alias) + '/root/nodes?descendentlevels=' + str(depth)
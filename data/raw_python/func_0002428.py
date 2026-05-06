def merge_wheres(self, wheres, bindings):
        """
        Merge a list of where clauses and bindings

        :param wheres: A list of where clauses
        :type wheres: list

        :param bindings: A list of bindings
        :type bindings: list

        :rtype: None
        """
        self.wheres = self.wheres + wheres
        self._bindings['where'] = self._bindings['where'] + bindings
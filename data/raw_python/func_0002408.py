def _set_join(self, query=None):
        """
        Set the join clause for the relation query.

        :param query: The query builder
        :type query: eloquent.orm.Builder

        :return: self
        :rtype: BelongsToMany
        """
        if not query:
            query = self._query

        base_table = self._related.get_table()

        key = '%s.%s' % (base_table, self._related.get_key_name())

        query.join(self._table, key, '=', self.get_other_key())

        return self
def get_relation_count_query(self, query, parent):
        """
        Add the constraints for a relationship count query.

        :type query: Builder
        :type parent: Builder

        :rtype: Builder
        """
        parent_table = self._parent.get_table()

        self._set_join(query)

        query.select(QueryExpression('COUNT(*)'))

        key = self.wrap('%s.%s' % (parent_table, self._first_key))

        return query.where(self.get_has_compare_key(), '=', QueryExpression(key))
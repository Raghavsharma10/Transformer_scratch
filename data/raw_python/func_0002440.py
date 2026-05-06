def get_relation_count_query(self, query, parent):
        """
        Add the constraints for a relationship count query.

        :type query: eloquent.orm.Builder
        :type parent: eloquent.orm.Builder

        :rtype: Builder
        """
        query.select(QueryExpression('COUNT(*)'))

        other_key = self.wrap('%s.%s' % (query.get_model().get_table(), self._other_key))

        return query.where(self.get_qualified_foreign_key(), '=', QueryExpression(other_key))
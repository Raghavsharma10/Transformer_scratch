def get_relation_count_query_for_self_join(self, query, parent):
        """
        Add the constraints for a relationship count query on the same table.

        :type query: eloquent.orm.Builder
        :type parent: eloquent.orm.Builder

        :rtype: eloquent.orm.Builder
        """
        query.select(QueryExpression('COUNT(*)'))

        table_prefix = self._query.get_query().get_connection().get_table_prefix()

        hash_ = self.get_relation_count_hash()
        query.from_('%s AS %s%s' % (self._table, table_prefix, hash_))

        key = self.wrap(self.get_qualified_parent_key_name())

        return query.where('%s.%s' % (hash_, self._foreign_key), '=', QueryExpression(key))
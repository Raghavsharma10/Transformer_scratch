def _set_join(self, query=None):
        """
        Set the join clause for the query.
        """
        if not query:
            query = self._query

        foreign_key = '%s.%s' % (self._related.get_table(), self._second_key)

        query.join(self._parent.get_table(), self.get_qualified_parent_key_name(), '=', foreign_key)
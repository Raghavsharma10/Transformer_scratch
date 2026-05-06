def raw_query(self, query, query_parameters=None):
        """
        To get all the document that equal to the query

        @param str query: The rql query
        @param dict query_parameters: Add query parameters to the query {key : value}
        """
        self.assert_no_raw_query()

        if len(self._where_tokens) != 0 or len(self._select_tokens) != 0 or len(
                self._order_by_tokens) != 0 or len(self._group_by_tokens) != 0:
            raise InvalidOperationException(
                "You can only use raw_query on a new query, without applying any operations "
                "(such as where, select, order_by, group_by, etc)")

        if query_parameters:
            self.query_parameters = query_parameters
        self._query = query
        return self
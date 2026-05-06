def search(self, field_name, search_terms, operator=QueryOperator.OR):
        """
        For more complex text searching

        @param str field_name: The field name in the index you want to query.
        :type str
        @param str search_terms: The terms you want to query
        @param QueryOperator operator: OR or AND
        """

        if field_name is None:
            raise ValueError("None field_name is invalid")

        field_name = Query.escape_if_needed(field_name)
        self._add_operator_if_needed()
        self.negate_if_needed(field_name)

        self.last_equality = {field_name: "(" + search_terms + ")" if ' ' in search_terms else search_terms}
        token = _Token(field_name=field_name, token="search", value=self.add_query_parameter(search_terms),
                       search_operator=operator)
        token.write = self.rql_where_write(token)
        self._where_tokens.append(token)
        return self
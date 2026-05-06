def where_in(self, field_name, values, exact=False):
        """
        Check that the field has one of the specified values

        @param str field_name: Name of the field
        @param str values: The values we wish to query
        @param bool exact: Getting the exact query (ex. case sensitive)
        """
        field_name = Query.escape_if_needed(field_name)
        self._add_operator_if_needed()
        self.negate_if_needed(field_name)

        token = _Token(field_name=field_name, value=self.add_query_parameter(list(Utils.unpack_iterable(values))),
                       token="in", exact=exact)
        token.write = self.rql_where_write(token)
        self._where_tokens.append(token)

        return self
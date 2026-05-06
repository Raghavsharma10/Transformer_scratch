def where_ends_with(self, field_name, value):
        """
        To get all the document that ends with the value in the giving field_name

        @param str field_name:The field name in the index you want to query.
        @param str value: The value will be the fields value you want to query
        """
        if field_name is None:
            raise ValueError("None field_name is invalid")

        field_name = Query.escape_if_needed(field_name)
        self._add_operator_if_needed()
        self.negate_if_needed(field_name)

        self.last_equality = {field_name: value}
        token = _Token(field_name=field_name, token="endsWith", value=self.add_query_parameter(value))
        token.write = self.rql_where_write(token)
        self._where_tokens.append(token)

        return self
def where_equals(self, field_name, value, exact=False):
        """
        To get all the document that equal to the value in the given field_name

        @param str field_name: The field name in the index you want to query.
        @param value: The value will be the fields value you want to query
        @param bool exact: If True getting exact match of the query
        """
        if field_name is None:
            raise ValueError("None field_name is invalid")

        field_name = Query.escape_if_needed(field_name)
        self._add_operator_if_needed()

        token = "equals"
        if self.negate:
            self.negate = False
            token = "not_equals"

        self.last_equality = {field_name: value}
        token = _Token(field_name=field_name, value=self.add_query_parameter(value), token=token, exact=exact)
        token.write = self.rql_where_write(token)
        self._where_tokens.append(token)

        return self
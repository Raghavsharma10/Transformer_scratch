def _escape_query(self, query, escaped=False):
        """
        Escapes query if it's not already escaped.

        Args:
            query: Query value.
            escaped (bool): expresses if query already escaped or not.

        Returns:
            Escaped query value.
        """
        if escaped:
            return query
        query = six.text_type(query)
        for e in ['+', '-', '&&', '||', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*',
                  '?', ':', ' ']:
            query = query.replace(e, "\\%s" % e)
        return query
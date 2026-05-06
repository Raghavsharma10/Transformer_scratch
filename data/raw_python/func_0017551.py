def dmql(query):
        """Client supplied raw DMQL, ensure quote wrap."""
        if isinstance(query, dict):
            raise ValueError("You supplied a dictionary to the dmql_query parameter, but a string is required."
                                " Did you mean to pass this to the search_filter parameter? ")

        # automatically surround the given query with parentheses if it doesn't have them already
        if len(query) > 0 and query != "*" and query[0] != '(' and query[-1] != ')':
            query = '({})'.format(query)
        return query
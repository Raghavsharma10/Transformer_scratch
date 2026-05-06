def _process_queries(self, queries):
        """Takes a list of queries and returns query clause value

        :arg queries: list of Q instances

        :returns: dict which is the query clause value

        """
        # First, let's mush everything into a single Q. Then we can
        # parse that into bits.
        new_q = Q()

        for query in queries:
            new_q += query

        # Now we have a single Q that needs to be processed.
        should_q = [self._process_query(query) for query in new_q.should_q]
        must_q = [self._process_query(query) for query in new_q.must_q]
        must_not_q = [self._process_query(query) for query in new_q.must_not_q]

        if len(must_q) > 1 or (len(should_q) + len(must_not_q) > 0):
            # If there's more than one must_q or there are must_not_q
            # or should_q, then we need to wrap the whole thing in a
            # boolean query.
            bool_query = {}
            if must_q:
                bool_query['must'] = must_q
            if should_q:
                bool_query['should'] = should_q
            if must_not_q:
                bool_query['must_not'] = must_not_q
            return {'bool': bool_query}

        if must_q:
            # There's only one must_q query and that's it, so we hoist
            # that.
            return must_q[0]

        return {}
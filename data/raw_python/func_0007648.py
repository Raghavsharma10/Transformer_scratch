def limit(self, maximum):
        """
        Return a new query, limited to a certain number of results.

        Unlike core reporting queries, you cannot specify a starting
        point for live queries, just the maximum results returned.

        ```python
        # first 50
        query.limit(50)
        ```
        """

        self.meta['limit'] = maximum
        self.raw.update({
            'max_results': maximum,
        })
        return self
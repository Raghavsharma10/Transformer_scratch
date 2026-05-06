def limit(self, *_range):
        """
        Return a new query, limited to a certain number of results.

        ```python
        # first 100
        query.limit(100)
        # 50 to 60
        query.limit(50, 10)
        ```

        Please note carefully that Google Analytics uses
        1-indexing on its rows.
        """

        # uses the same argument order as
        # LIMIT in a SQL database
        if len(_range) == 2:
            start, maximum = _range
        else:
            start = 1
            maximum = _range[0]

        self.meta['limit'] = maximum

        self.raw.update({
            'start_index': start,
            'max_results': maximum,
        })
        return self
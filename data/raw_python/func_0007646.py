def next(self):
        """
        Return a new query with a modified `start_index`.
        Mainly used internally to paginate through results.
        """
        step = self.raw.get('max_results', 1000)
        start = self.raw.get('start_index', 1) + step
        self.raw['start_index'] = start
        return self
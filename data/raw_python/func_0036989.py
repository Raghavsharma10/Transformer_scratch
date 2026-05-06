def set(self, results):
        """Set results.

        results is an iterable of tuples, where each tuple is a row of results.

        >>> x = Results(['title'])
        >>> x.set([('Konosuba',), ('Oreimo',)])
        >>> x
        Results(['title'], [('Konosuba',), ('Oreimo',)])

        """
        self.results = list()
        for row in results:
            self.append(row)
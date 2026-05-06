def append(self, row):
        """Append a result row and check its length.

        >>> x = Results(['title', 'type'])
        >>> x.append(('Konosuba', 'TV'))
        >>> x
        Results(['title', 'type'], [('Konosuba', 'TV')])

        >>> x.append(('Konosuba',))
        Traceback (most recent call last):
            ...
        ValueError: Wrong result row length

        """
        row = tuple(row)
        if len(row) != self.table_width:
            raise ValueError('Wrong result row length')
        self.results.append(row)
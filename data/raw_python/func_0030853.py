def sum(self, axis=0, dtype=None):
        """Sum over rows or columns.
        
        Overrides the default `pandas.DataFrame.sum()` function to prevent
        temporary in-memory copies.
        """
        if axis not in [0, 1, 'index', 'columns']:
            raise ValueError('"axis" parameter must be one of 0, 1, "index", '
                             'or "columns".')

        sum_kwargs = {}
        if dtype is not None:
            sum_kwargs['dtype'] = dtype
        y = self.values.sum(axis=axis, **sum_kwargs)

        if axis == 0 or axis == 'index':
            y = profile.ExpProfile(y, genes=self.cells)
        else:
            y = profile.ExpProfile(y, genes=self.genes)

        return y
def apply(self, func, window=None, bycolumn=True, align=None, **kwargs):
        '''Apply function ``func`` to the timeseries.

        :keyword func: string indicating function to apply
        :keyword window: Rolling window, If not defined ``func`` is applied on
            the whole dataset. Default ``None``.
        :keyword bycolumn: If ``True``, function ``func`` is applied on
            each column separately. Default ``True``.
        :keyword align: string specifying whether the index of the result
            should be ``left`` or ``right`` (default) or ``centered``
            aligned compared to the rolling window of observations.
        :keyword kwargs: dictionary of auxiliary parameters used by
            function ``func``.
        '''
        N = len(self)
        window = window or N
        self.precondition(window <= N and window > 0, OutOfBound)
        return self._rollapply(func,
                               window=window,
                               align=align or self.default_align,
                               bycolumn=bycolumn,
                               **kwargs)
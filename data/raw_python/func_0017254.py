def preprocess(self, ticker, start, end, logger, backend, **kwargs):
        '''Preprocess **hook**. This is first loading hook and it is
**called before requesting data** from a dataprovider.
It must return an instance of :attr:`TimeSerieLoader.preprocessdata`.
By default it returns::

    self.preprocessdata(intervals = ((start,end),))

It could be overritten to modify the intervals.
If the intervals is ``None`` or an empty container,
the :func:`dynts.data.DataProvider.load` method won't be called,
otherwise it will be called as many times as the number of intervals
in the return tuple (by default once).
'''
        return self.preprocessdata(intervals = ((start, end),))
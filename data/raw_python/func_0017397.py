def reduce(self, size, method='simple', **kwargs):
        '''Trim :class:`Timeseries` to a new *size* using the algorithm
*method*. If *size* is greater or equal than len(self) it does nothing.'''
        if size >= len(self):
            return self
        return self.getalgo('reduce', method)(self, size, **kwargs)
def dates(self, desc=None):
        '''Returns an iterable over ``datetime.date`` instances
in the timeseries.'''
        c = self.dateinverse
        for key in self.keys(desc=desc):
            yield c(key)
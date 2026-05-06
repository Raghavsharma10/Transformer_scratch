def clone(self, date=None, data=None, name=None):
        '''Create a clone of timeseries'''
        name = name or self.name
        data = data if data is not None else self.values()
        ts = self.__class__(name)
        ts._dtype = self._dtype
        if date is None:
            # dates not provided
            ts.make(self.keys(), data, raw=True)
        else:
            ts.make(date, data)
        return ts
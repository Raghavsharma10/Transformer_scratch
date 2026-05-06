def insert(self, dte, values):
        '''insert *values* at date *dte*.'''
        if len(values):
            dte = self.dateconvert(dte)
            if not self:
                self._date = np.array([dte])
                self._data = np.array([values])
            else:
                # search for the date
                index = self._skl.rank(dte)
                if index < 0:
                    # date not available
                    N = len(self._data)
                    index = -1-index
                    self._date.resize((N+1,))
                    self._data.resize((N+1, self.count()))
                    if index < N:
                        self._date[index+1:] = self._date[index:-1]
                        self._data[index+1:] = self._data[index:-1]
                self._date[index] = dte
                self._data[index] = values
            self._skl.insert(dte)
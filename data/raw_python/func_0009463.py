def addDarkCurrent(self, slope, intercept=None, date=None, info='', error=None):
        '''
        Args:
            slope (np.array)
            intercept (np.array)
            error (numpy.array)
            slope (float): dPx/dExposureTime[sec]
            error (float): absolute
            date (str): "DD Mon YY" e.g. "30 Nov 16"
        '''
        date = _toDate(date)

        self._checkShape(slope)
        self._checkShape(intercept)

        d = self.coeffs['dark current']
        if intercept is None:
            data = slope
        else:
            data = (slope, intercept)
        d.insert(_insertDateIndex(date, d), [date, info, data, error])
def check_in_bounds(self, date):
        '''Check that left and right bounds are sane

        :param date: date to validate left/right bounds for
        '''
        dt = Timestamp(date)
        return ((self._lbound is None or dt >= self._lbound) and
                (self._rbound is None or dt <= self._rbound))
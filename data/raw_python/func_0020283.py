def intervals(self, startdate, enddate, parseinterval=None):
        '''Given a ``startdate`` and an ``enddate`` dates, evaluate the
        date intervals from which data is not available. It return a list
        of two-dimensional tuples containing start and end date for the
        interval. The list could contain 0, 1 or 2 tuples.'''
        return missing_intervals(startdate, enddate, self.data_start,
                                 self.data_end, dateconverter=self.todate,
                                 parseinterval=parseinterval)
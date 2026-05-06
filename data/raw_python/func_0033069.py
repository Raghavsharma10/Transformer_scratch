def started_after(self, date):
        '''
        Leaves only those objects whose first version started after the
        specified date.

        :param date: date string to use in calculation
        '''
        dt = Timestamp(date)
        starts = self.groupby(self._oid).apply(lambda df: df._start.min())
        oids = set(starts[starts > dt].index.tolist())
        return self[self._oid.apply(lambda v: v in oids)]
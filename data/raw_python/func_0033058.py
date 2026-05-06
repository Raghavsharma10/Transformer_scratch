def on_date(self, date, only_count=False):
        '''
        Filters out only the rows that match the spectified date.
        Works only on a Result that has _start and _end columns.

        :param date: date can be anything Pandas.Timestamp supports parsing
        :param only_count: return back only the match count
        '''
        if not self.check_in_bounds(date):
            raise ValueError('Date %s is not in the queried range.' % date)
        date = Timestamp(date)
        after_start = self._start <= date
        before_end = (self._end > date) | self._end_isnull
        if only_count:
            return np.sum(before_end & after_start)
        else:
            return self.filter(before_end & after_start)
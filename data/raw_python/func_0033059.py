def history(self, dates=None, linreg_since=None, lin_reg_days=20):
        '''
        Works only on a Result that has _start and _end columns.

        :param dates: list of dates to query
        :param linreg_since: estimate future values using linear regression.
        :param lin_reg_days: number of past days to use as prediction basis
        '''
        dates = dates or self.get_dates_range()
        vals = [self.on_date(dt, only_count=True) for dt in dates]
        ret = Series(vals, index=dates)
        if linreg_since is not None:
            ret = self._linreg_future(ret, linreg_since, lin_reg_days)
        return ret.sort_index()
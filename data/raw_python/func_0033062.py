def _auto_select_scale(self, start=None, end=None, ideal=300):
        '''
        Guess what a good timeseries scale might be,
        given a particular data set, attempting to
        make the total number of x values as close to
        `ideal` as possible

        This is a helper for plotting
        '''
        start = start or self._start.min()
        end = end or max(self._end.max(), self._start.max())
        daily_count = (end - start).days
        if daily_count <= ideal:
            return 'daily'
        elif daily_count / 7 <= ideal:
            return 'weekly'
        elif daily_count / 30 <= ideal:
            return 'monthly'
        elif daily_count / 91 <= ideal:
            return 'quarterly'
        else:
            return 'yearly'
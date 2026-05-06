def get_dates_range(self, scale='auto', start=None, end=None,
                        date_max='2010-01-01'):
        '''
        Returns a list of dates sampled according to the specified parameters.

        :param scale: {'auto', 'maximum', 'daily', 'weekly', 'monthly',
            'quarterly', 'yearly'}
            Scale specifies the sampling intervals.
            'auto' will heuristically choose a scale for quick processing
        :param start: First date that will be included.
        :param end: Last date that will be included
        '''
        if scale not in ['auto', 'maximum', 'daily', 'weekly', 'monthly',
                         'quarterly', 'yearly']:
            raise ValueError('Incorrect scale: %s' % scale)
        start = Timestamp(start or self._start.min() or date_max)
        # FIXME: start != start is true for NaN objects... is NaT the same?
        start = Timestamp(date_max) if repr(start) == 'NaT' else start
        end = Timestamp(end or max(Timestamp(self._end.max()),
                                   self._start.max()))
        # FIXME: end != end ?
        end = datetime.utcnow() if repr(end) == 'NaT' else end
        start = start if self.check_in_bounds(start) else self._lbound
        end = end if self.check_in_bounds(end) else self._rbound

        if scale == 'auto':
            scale = self._auto_select_scale(start, end)
        if scale == 'maximum':
            start_dts = list(self._start.dropna().values)
            end_dts = list(self._end.dropna().values)
            dts = map(Timestamp, set(start_dts + end_dts))
            dts = filter(lambda ts: self.check_in_bounds(ts) and
                         ts >= start and ts <= end, dts)
            return dts

        freq = dict(daily='D', weekly='W', monthly='M', quarterly='3M',
                    yearly='12M')
        offset = dict(daily=off.Day(n=0), weekly=off.Week(),
                      monthly=off.MonthEnd(), quarterly=off.QuarterEnd(),
                      yearly=off.YearEnd())
        # for some reason, weekly date range gives one week less:
        end_ = end + off.Week() if scale == 'weekly' else end
        ret = list(pd.date_range(start + offset[scale], end_,
                                 freq=freq[scale]))
        ret = [dt for dt in ret if dt <= end]
        ret = [start] + ret if ret and start < ret[0] else ret
        ret = ret + [end] if ret and end > ret[-1] else ret
        ret = filter(lambda ts: self.check_in_bounds(ts), ret)
        return ret
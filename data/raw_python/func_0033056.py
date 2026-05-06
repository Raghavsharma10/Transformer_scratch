def set_date_bounds(self, date):
        '''
        Pass in the date used in the original query.

        :param date: Date (date range) that was queried:
            date -> 'd', '~d', 'd~', 'd~d'
            d -> '%Y-%m-%d %H:%M:%S,%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'
        '''
        if date is not None:
            split = date.split('~')
            if len(split) == 1:
                self._lbound = ts2dt(date)
                self._rbound = ts2dt(date)
            elif len(split) == 2:
                if split[0] != '':
                    self._lbound = ts2dt(split[0])
                if split[1] != '':
                    self._rbound = ts2dt(split[1])
            else:
                raise Exception('Date %s is not in the correct format' % date)
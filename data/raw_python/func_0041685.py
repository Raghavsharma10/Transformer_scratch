def _ndays(self, start_date, ndays):
        """
        Compute an end date given a start date and a number of days.
        """
        if not getattr(self.args, 'start-date') and not self.config.get('start-date', None):
            raise Exception('start-date must be provided when ndays is used.')

        d = date(*map(int, start_date.split('-')))
        d += timedelta(days=ndays)

        return d.strftime('%Y-%m-%d')
def stats_send(self, template, start_date, end_date, options=None):
        """
        Retrieve information about a particular transactional or aggregated information
        from transactionals from that template over a specified date range.
        http://docs.sailthru.com/api/stat
        """
        options = options or {}
        data = options.copy()
        data = {'template': template,
                'start_date': start_date,
                'end_date': end_date
                }
        data['stat'] = 'send'
        return self._stats(data)
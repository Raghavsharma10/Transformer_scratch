def stats_blast(self, blast_id=None, start_date=None, end_date=None, options=None):
        """
        Retrieve information about a particular blast or aggregated information from all of blasts over a specified date range.
        http://docs.sailthru.com/api/stat
        """
        options = options or {}
        data = options.copy()
        if blast_id is not None:
            data['blast_id'] = blast_id
        if start_date is not None:
            data['start_date'] = start_date
        if end_date is not None:
            data['end_date'] = end_date
        data['stat'] = 'blast'
        return self._stats(data)
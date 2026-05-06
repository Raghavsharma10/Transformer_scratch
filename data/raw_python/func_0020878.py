def stats_list(self, list=None, date=None, headers=None):
        """
        Retrieve information about your subscriber counts on a particular list, on a particular day.
        http://docs.sailthru.com/api/stat
        """
        data = {'stat': 'list'}
        if list is not None:
            data['list'] = list
        if date is not None:
            data['date'] = date
        return self._stats(data, headers)
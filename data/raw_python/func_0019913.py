def search(self, pattern, start=None, limit=None, include_category=None):
        '''Searches for domains that match a given pattern'''

        params = dict()

        if start is None:
            start = datetime.timedelta(days=30)

        if isinstance(start, datetime.timedelta):
            params['start'] = int(time.mktime((datetime.datetime.utcnow() - start).timetuple()) * 1000)
        elif isinstance(start, datetime.datetime):
            params['start'] = int(time.mktime(start.timetuple()) * 1000)
        else:
            raise Investigate.SEARCH_ERR

        if limit is not None and isinstance(limit, int):
            params['limit'] = limit
        if include_category is not None and isinstance(include_category, bool):
            params['includeCategory'] = str(include_category).lower()

        uri = self._uris['search'].format(quote_plus(pattern))

        return self.get_parse(uri, params)
def info(self, url, limit=None):
        """GETs "info" about ``url``.  See https://github.com/reddit/reddit/wiki/API%3A-info.json.
        
        URL: ``http://www.reddit.com/api/info/?url=<url>``
        
        :param url: url
        :param limit: max number of links to get
        """
        return self._limit_get('api', 'info', params=dict(url=url), limit=limit)
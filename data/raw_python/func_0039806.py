def search(self, query, limit=None):
        """Use reddit's search function.  Returns :class:`things.Listing` object.
        
        URL: ``http://www.reddit.com/search/?q=<query>&limit=<limit>``
        
        :param query: query string
        :param limit: max number of results to get
        """
        return self._limit_get('search', params=dict(q=query), limit=limit)
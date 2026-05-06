def flairlist(self, r, limit=1000, after=None, before=None):
        """Login required.  Gets flairlist for subreddit `r`.  See https://github.com/reddit/reddit/wiki/API%3A-flairlist.
        
        However, the wiki docs are wrong (as of 2012/5/4).  Returns :class:`things.ListBlob` of :class:`things.Blob` objects, each object being a mapping with `user`, `flair\_css\_class`, and `flair\_text` attributes.
        
        URL: ``http://www.reddit.com/r/<r>/api/flairlist``
        
        :param r: name of subreddit
        :param limit: max number of items to return
        :param after: full id of user to return entries after
        :param before: full id of user to return entries *before* 
        """
        params = dict(limit=limit)
        if after:
            params['after'] = after
        elif before:
            params['before'] = before
        b = self.get('r', r, 'api', 'flairlist', params=params)
        return b.users
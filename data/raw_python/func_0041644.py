def flairlist(self, limit=1000, after=None, before=None):
        """GETs flairlist for this subreddit.  Calls :meth:`narwal.Reddit.flairlist`.
        
        :param limit: max number of items to return
        :param after: full id of user to return entries after
        :param before: full id of user to return entries *before*
        """
        return self._reddit.flairlist(self.display_name, limit=limit, after=after, before=before)
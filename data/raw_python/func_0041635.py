def hot(self, limit=None):
        """GETs hot links from this subreddit.  Calls :meth:`narwal.Reddit.hot`.
        
        :param limit: max number of links to return
        """
        return self._reddit.hot(self.display_name, limit=limit)
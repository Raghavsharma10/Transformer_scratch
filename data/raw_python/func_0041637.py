def top(self, limit=None):
        """GETs top links from this subreddit.  Calls :meth:`narwal.Reddit.top`.
        
        :param limit: max number of links to return
        """
        return self._reddit.top(self.display_name, limit=limit)
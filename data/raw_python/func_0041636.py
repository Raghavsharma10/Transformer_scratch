def new(self, limit=None):
        """GETs new links from this subreddit.  Calls :meth:`narwal.Reddit.new`.
        
        :param limit: max number of links to return
        """
        return self._reddit.new(self.display_name, limit=limit)
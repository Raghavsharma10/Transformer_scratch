def controversial(self, limit=None):
        """GETs controversial links from this subreddit.  Calls :meth:`narwal.Reddit.controversial`.
        
        :param limit: max number of links to return
        """
        return self._reddit.controversial(self.display_name, limit=limit)
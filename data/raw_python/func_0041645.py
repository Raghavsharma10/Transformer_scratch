def contributors(self, limit=None):
        """GETs contributors for this subreddit.  Calls :meth:`narwal.Reddit.contributors`.
        
        :param limit: max number of items to return
        """
        return self._reddit.contributors(self.display_name, limit=limit)
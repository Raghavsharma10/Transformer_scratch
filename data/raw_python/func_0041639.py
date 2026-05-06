def comments(self, limit=None):
        """GETs newest comments from this subreddit.  Calls :meth:`narwal.Reddit.comments`.
        
        :param limit: max number of links to return
        """
        return self._reddit.comments(self.display_name, limit=limit)
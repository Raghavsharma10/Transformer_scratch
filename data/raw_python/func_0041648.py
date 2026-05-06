def comments(self, limit=None):
        """GETs user's comments.  Calls :meth:`narwal.Reddit.user_comments`.
        
        :param limit: max number of comments to get
        """
        return self._reddit.user_comments(self.name, limit=limit)
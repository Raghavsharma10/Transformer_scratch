def moderators(self, limit=None):
        """GETs moderators for this subreddit.  Calls :meth:`narwal.Reddit.moderators`.
        
        :param limit: max number of items to return
        """
        return self._reddit.moderators(self.display_name, limit=limit)
def overview(self, limit=None):
        """GETs overview of user's activities.  Calls :meth:`narwal.Reddit.user_overview`.
        
        :param limit: max number of items to get
        """
        return self._reddit.user_overview(self.name, limit=limit)
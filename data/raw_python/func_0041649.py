def submitted(self, limit=None):
        """GETs user's submissions.  Calls :meth:`narwal.Reddit.user_submitted`.
        
        :param limit: max number of submissions to get
        """
        return self._reddit.user_submitted(self.name, limit=limit)
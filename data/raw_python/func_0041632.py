def distinguish(self, how=True):
        """Distinguishes this thing (POST).  Calls :meth:`narwal.Reddit.distinguish`.
        
        :param how: either True, False, or 'admin'
        """
        return self._reddit.distinguish(self.name, how=how)
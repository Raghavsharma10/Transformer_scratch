def comments(self, limit=None):
        """GETs comments to this thing.
        
        :param limit: max number of comments to return
        """
        return self._reddit._limit_get(self.permalink, limit=limit)[1]
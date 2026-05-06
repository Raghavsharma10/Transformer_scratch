def cloneQuery(self, limit=_noItem, sort=_noItem):
        """
        Clone the original query which this distinct query wraps, and return a new
        wrapper around that clone.
        """
        newq = self.query.cloneQuery(limit=limit, sort=sort)
        return self.__class__(newq)
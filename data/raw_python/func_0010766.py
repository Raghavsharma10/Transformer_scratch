def last(self, n=10, by=None, **kwargs):
        """
        Alias for .tail().
        """
        return self.tail(n=n, by=by, **kwargs)
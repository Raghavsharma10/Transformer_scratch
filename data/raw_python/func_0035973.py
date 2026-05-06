def readlines(self, *args, **kwargs):
        """Return list of all lines. Always returns list of unicode."""
        return list(iter(partial(self.readline, *args, **kwargs), u''))
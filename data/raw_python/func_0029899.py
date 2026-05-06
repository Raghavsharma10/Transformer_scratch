def cache_key(self):
        """The name in a form suitable for use as a cache-key"""
        try:
            return self.path
        except TypeError:
            raise TypeError("self.path is invalild: '{}', '{}'".format(str(self.path), type(self.path)))
def get(self, key, failobj=None, exact=0):

        """Returns failobj if key is not found or is ambiguous"""

        if not exact:
            try:
                key = self.getfullkey(key)
            except KeyError:
                return failobj
        return self.data.get(key,failobj)
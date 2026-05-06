def get(self, key, failobj=None, exact=0):
        """Raises exception if key is ambiguous"""
        if not exact:
            key = self.getfullkey(key,new=1)
        return self.data.get(key,failobj)
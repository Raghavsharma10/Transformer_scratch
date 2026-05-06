def get(self, copy=False):
        """Return the value of the attribute"""
        array = getattr(self.owner, self.name)
        if copy:
            return array.copy()
        else:
            return array
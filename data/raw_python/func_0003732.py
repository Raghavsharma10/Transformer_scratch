def get(self, subset=None):
        """Return a dictionary object with the registered fields and their values

           Optional rgument:
            | ``subset``  --  a list of names to restrict the number of fields
                              in the result
        """
        if subset is None:
            return dict((name, attr.get(copy=True)) for name, attr in self._fields.items())
        else:
            return dict((name, attr.get(copy=True)) for name, attr in self._fields.items() if name in subset)
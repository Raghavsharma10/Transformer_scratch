def _reset(self, **kwargs):
        """
        Reset after repopulating from API (or when initializing).
        """
        # set object attributes from params
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # set defaults (if need be) where the default is not None
        for attr in self.ATTRIBUTES:
            if not hasattr(self, attr) and self.ATTRIBUTES[attr] is not None:
                setattr(self, attr, self.ATTRIBUTES[attr])
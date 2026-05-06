def instantiate(self):
        """
        Instantiate this object with the supplied parameters in `self.kwds`,
        or if already instantiated, return the cached instance.
        """
        if self.instance is None:
            self.instance = checked_call(self.cls, self.kwds)
        #endif
        try:
            self.instance.yaml_src = self.yaml_src
        except AttributeError:
            pass
        return self.instance
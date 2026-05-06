def keys(self):
        """Return a merged set of top level keys from all configurations."""
        s = set()
        for config in self.__configs:
            s |= config.keys()
        return s
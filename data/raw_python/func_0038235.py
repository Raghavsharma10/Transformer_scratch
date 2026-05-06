def getmembers(self):
        """
        :return: list of members as name, type tuples
        :rtype: list
        """
        return filter(
            lambda m: not m[0].startswith("__") and not inspect.isfunction(m[1]) and not inspect.ismethod(m[1]),
            inspect.getmembers(self.__class__)
        )
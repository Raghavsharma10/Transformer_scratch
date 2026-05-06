def getConfigurableParent(cls):
        """
        Return the parent from which this class inherits configurations
        """
        for p in cls.__bases__:
            if isinstance(p, Configurable) and p is not Configurable:
                return p
        return None
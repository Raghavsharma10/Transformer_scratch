def getConfigRoot(cls, create = False):
        """
        Return the mapped configuration root node
        """
        try:
            return manager.gettree(getattr(cls, 'configkey'), create)
        except AttributeError:
            return None
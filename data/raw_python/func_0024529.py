def _get_subclass(name):
        """
        need for cyclic import solving
        """
        return next(x for x in BaseContainer.__subclasses__() if x.__name__ == name)
def register(cls, name):
        """ Decorator to register the event identified by `name`.

        Return the decorated class.

        Raise GerritError if the event is already registered.

        """

        def decorate(klazz):
            """ Decorator. """
            if name in cls._events:
                raise GerritError("Duplicate event: %s" % name)
            cls._events[name] = [klazz.__module__, klazz.__name__]
            klazz.name = name
            return klazz
        return decorate
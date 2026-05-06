def with_indices(*args):
    '''
    Create indices for an event class. Every event class must be decorated with this decorator.
    '''
    def decorator(cls):
        for c in cls.__bases__:
            if hasattr(c, '_indicesNames'):
                cls._classnameIndex = c._classnameIndex + 1
                for i in range(0, cls._classnameIndex):
                    setattr(cls, '_classname' + str(i), getattr(c, '_classname' + str(i)))
                setattr(cls, '_classname' + str(cls._classnameIndex), cls._getTypename())
                cls._indicesNames = c._indicesNames + ('_classname' + str(cls._classnameIndex),) + args
                cls._generateTemplate()
                return cls
        cls._classnameIndex = -1
        cls._indicesNames = args
        cls._generateTemplate()
        return cls
    return decorator
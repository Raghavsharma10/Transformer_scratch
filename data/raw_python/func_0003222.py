def getTypename(cls):
        '''
        :returns: return the proper name to match
        '''
        if cls is Event:
            return None
        else:
            for c in cls.__bases__:
                if issubclass(c, Event):
                    if c is Event:
                        return cls._getTypename()
                    else:
                        return c.getTypename()
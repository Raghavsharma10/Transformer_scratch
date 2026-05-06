def fromDict(cls, d):
        """
        Create a new instance from attribute values provided in a dictionary.

        @param d: A C{dict} with keys/values for the attributes of a new
            instance of this class. Keys 'id' and 'sequence' with C{str} values
            must be provided. A 'quality' C{str} key is optional. Keys 'start'
            and 'stop' must have C{int} values. Keys 'openLeft' and 'openRight'
            are C{bool}, all keys are as described in the docstring for this
            class.
        @return: A new instance of this class, with values taken from C{d}.
        """
        # Make a dummy instance whose attributes we can set explicitly.
        new = cls(AARead('', ''), 0, 0, True, True)
        new.id = d['id']
        new.sequence = d['sequence']
        new.quality = d.get('quality')
        new.start = d['start']
        new.stop = d['stop']
        new.openLeft = d['openLeft']
        new.openRight = d['openRight']
        return new
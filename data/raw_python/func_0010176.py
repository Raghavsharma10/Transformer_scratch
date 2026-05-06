def _convert(cls, other, ignoreScalars=False):
        '''
        :other: Point or point equivalent
        :ignorescalars: optional boolean
        :return: Point

        Class private method for converting 'other' into a Point
        subclasss. If 'other' already is a Point subclass, nothing
        is done. If ignoreScalars is True and other is a float or int
        type, a TypeError exception is raised.
        '''
        if ignoreScalars:
            if isinstance(other, (int, float)):
                msg = "unable to convert {} to {}".format(other, cls.__name__)
                raise TypeError(msg)

        return cls(other) if not issubclass(type(other), cls) else other
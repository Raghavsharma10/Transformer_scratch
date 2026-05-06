def dump(cls):
        """Dump data into a dict.

        .. versionadded:: 0.0.2
        """
        d = OrderedDict(cls.Items())
        d["__classname__"] = cls.__name__
        for attr, klass in cls.Subclasses():
            d[attr] = klass.dump()
        return OrderedDict([(cls.__name__, d)])
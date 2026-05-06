def attributes(cls):
        """
        yields tuples for all attributes defined on this handler

        tuple yielded:
            name (str), attribute (Attribute)
        """

        for k in dir(cls):
            v = getattr(cls, k)
            if isinstance(v, Attribute):
                yield k,v
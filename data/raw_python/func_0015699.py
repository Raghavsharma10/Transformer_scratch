def _new_type(cls, args):
        """Creates a new class similar to namedtuple.

        Pass a list of field names or None for no field name.

        >>> x = ResultTuple._new_type([None, "bar"])
        >>> x((1, 3))
        ResultTuple(1, bar=3)
        """

        fformat = ["%r" if f is None else "%s=%%r" % f for f in args]
        fformat = "(%s)" % ", ".join(fformat)

        class _ResultTuple(cls):
            __slots__ = ()
            _fformat = fformat
            if args:
                for i, a in enumerate(args):
                    if a is not None:
                        vars()[a] = property(itemgetter(i))
                del i, a

        return _ResultTuple
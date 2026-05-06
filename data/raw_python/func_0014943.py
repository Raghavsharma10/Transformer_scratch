def mkRepr(instance, *argls, **kwargs):
    r"""Convinience function to implement ``__repr__``. `kwargs` values are
        ``repr`` ed. Special behavior for ``instance=None``: just the
        arguments are formatted.

    Example:

        >>> class Thing:
        ...     def __init__(self, color, shape, taste=None):
        ...         self.color, self.shape, self.taste = color, shape, taste
        ...     def __repr__(self):
        ...         return mkRepr(self, self.color, self.shape, taste=self.taste)
        ...
        >>> maggot = Thing('white', 'cylindrical', 'chicken')
        >>> maggot
        Thing('white', 'cylindrical', taste='chicken')
        >>> Thing('Color # 132942430-214809804-412430988081-241234', 'unkown', taste=maggot)
        Thing('Color # 132942430-214809804-412430988081-241234',
              'unkown',
              taste=Thing('white', 'cylindrical', taste='chicken'))
    """
    width=79
    maxIndent=15
    minIndent=2
    args = (map(repr, argls) + ["%s=%r" % (k, v)
                               for (k,v) in sorted(kwargs.items())]) or [""]
    if instance is not None:
        start = "%s(" % instance.__class__.__name__
        args[-1] += ")"
    else:
        start = ""
    if len(start) <= maxIndent and len(start) + len(args[0]) <= width and \
           max(map(len,args)) <= width: # XXX mag of last condition bit arbitrary
        indent = len(start)
        args[0] = start + args[0]
        if sum(map(len, args)) + 2*(len(args) - 1) <= width:
            return ", ".join(args)
    else:
        indent = minIndent
        args[0] = start + "\n" + " " * indent + args[0]
    return (",\n" + " " * indent).join(args)
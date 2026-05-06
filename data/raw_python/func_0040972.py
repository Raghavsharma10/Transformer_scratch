def _repr(self, *args, **kwargs):
        """Return a __repr__ string from the arguments provided to __init__.

        @param args: list of arguments to __init__
        @param kwargs: dictionary of keyword arguments to __init__
        @return: __repr__ string

        """
        # Remove unnecessary empty keywords arguments and sort the arguments
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs = OrderedDict(sorted(kwargs.items()))

        # Build the __repr__ string pieces
        args_repr = ', '.join(repr(arg) for arg in args)
        kwargs_repr = ', '.join(k + '=' + repr(v) for k, v in kwargs.items())
        if args_repr and kwargs_repr:
            kwargs_repr = ', ' + kwargs_repr
        name = self.__class__.__name__

        return "{}({}{})".format(name, args_repr, kwargs_repr)
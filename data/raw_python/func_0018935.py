def __getiterable(value):   # ToDo: refactor
        """Try to convert the given argument to a |list| of  |Selection|
        objects and return it.
        """
        if isinstance(value, Selection):
            return [value]
        try:
            for selection in value:
                if not isinstance(selection, Selection):
                    raise TypeError
            return list(value)
        except TypeError:
            raise TypeError(
                f'Binary operations on Selections objects are defined for '
                f'other Selections objects, single Selection objects, or '
                f'iterables containing `Selection` objects, but the type of '
                f'the given argument is `{objecttools.classname(value)}`.')
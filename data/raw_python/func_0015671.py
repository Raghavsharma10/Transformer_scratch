def _create(self, format, args):
        """Create a GVariant object from given format and argument list.

        This method recursively calls itself for complex structures (arrays,
        dictionaries, boxed).

        Return a tuple (variant, rest_format, rest_args) with the generated
        GVariant, the remainder of the format string, and the remainder of the
        arguments.

        If args is None, then this won't actually consume any arguments, and
        just parse the format string and generate empty GVariant structures.
        This is required for creating empty dictionaries or arrays.
        """
        # leaves (simple types)
        constructor = self._LEAF_CONSTRUCTORS.get(format[0])
        if constructor:
            if args is not None:
                if not args:
                    raise TypeError('not enough arguments for GVariant format string')
                v = constructor(args[0])
                return (v, format[1:], args[1:])
            else:
                return (None, format[1:], None)

        if format[0] == '(':
            return self._create_tuple(format, args)

        if format.startswith('a{'):
            return self._create_dict(format, args)

        if format[0] == 'a':
            return self._create_array(format, args)

        raise NotImplementedError('cannot handle GVariant type ' + format)
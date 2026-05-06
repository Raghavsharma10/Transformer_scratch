def _create_tuple(self, format, args):
        """Handle the case where the outermost type of format is a tuple."""

        format = format[1:]  # eat the '('
        if args is None:
            # empty value: we need to call _create() to parse the subtype
            rest_format = format
            while rest_format:
                if rest_format.startswith(')'):
                    break
                rest_format = self._create(rest_format, None)[1]
            else:
                raise TypeError('tuple type string not closed with )')

            rest_format = rest_format[1:]  # eat the )
            return (None, rest_format, None)
        else:
            if not args or not isinstance(args[0], tuple):
                raise TypeError('expected tuple argument')

            builder = GLib.VariantBuilder.new(variant_type_from_string('r'))
            for i in range(len(args[0])):
                if format.startswith(')'):
                    raise TypeError('too many arguments for tuple signature')

                (v, format, _) = self._create(format, args[0][i:])
                builder.add_value(v)
            args = args[1:]
            if not format.startswith(')'):
                raise TypeError('tuple type string not closed with )')

            rest_format = format[1:]  # eat the )
            return (builder.end(), rest_format, args)
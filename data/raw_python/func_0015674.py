def _create_array(self, format, args):
        """Handle the case where the outermost type of format is an array."""

        builder = None
        if args is None or not args[0]:
            # empty value: we need to call _create() to parse the subtype,
            # and specify the element type precisely
            rest_format = self._create(format[1:], None)[1]
            element_type = format[:len(format) - len(rest_format)]
            builder = GLib.VariantBuilder.new(variant_type_from_string(element_type))
        else:
            builder = GLib.VariantBuilder.new(variant_type_from_string('a*'))
            for i in range(len(args[0])):
                (v, rest_format, _) = self._create(format[1:], args[0][i:])
                builder.add_value(v)
        if args is not None:
            args = args[1:]
        return (builder.end(), rest_format, args)
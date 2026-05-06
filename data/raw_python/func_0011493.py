def _handle_constant(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_constant.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling constant type {}".format(node.type))
        switch = {
            "int": (self._str_to_int, self._choose_const_int_class),
            "long": (self._str_to_int, self._choose_const_int_class),
            # TODO this isn't quite right, but py010parser wouldn't have
            # parsed it if it wasn't correct...
            "float": (lambda x: float(x.lower().replace("f", "")), fields.Float),
            "double": (float, fields.Double),

            # cut out the quotes
            "char": (lambda x: ord(utils.string_escape(x[1:-1])), fields.Char),

            # TODO should this be unicode?? will probably bite me later...
            # cut out the quotes
            "string": (lambda x: str(utils.string_escape(x[1:-1])), fields.String)
        }

        if node.type in switch:
            #return switch[node.type](node.value)
            conversion,field_cls = switch[node.type]
            val = conversion(node.value)

            if hasattr(field_cls, "__call__") and not type(field_cls) is type:
                field_cls = field_cls(val)

            field = field_cls()
            field._pfp__set_value(val)
            return field

        raise UnsupportedConstantType(node.coord, node.type)
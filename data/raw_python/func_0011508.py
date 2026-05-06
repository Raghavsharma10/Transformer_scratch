def _handle_enum(self, node, scope, ctxt, stream):
        """Handle enum nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling enum")
        if node.type is None:
            enum_cls = fields.Int
        else:
            enum_cls = self._handle_node(node.type, scope, ctxt, stream)

        enum_vals = {}
        curr_val = enum_cls()
        curr_val._pfp__value = -1
        for enumerator in node.values.enumerators:
            if enumerator.value is not None:
                curr_val = self._handle_node(enumerator.value, scope, ctxt, stream)
            else:
                curr_val = curr_val + 1
            curr_val._pfp__freeze()
            enum_vals[enumerator.name] = curr_val
            enum_vals[fields.PYVAL(curr_val)] = enumerator.name
            scope.add_local(enumerator.name, curr_val)

        if node.name is not None:
            enum_cls = EnumDef(node.name, enum_cls, enum_vals)
            scope.add_type_class(node.name, enum_cls)

        else:
            enum_cls = EnumDef("enum_" + enum_cls.__name__, enum_cls, enum_vals)
            # don't add to scope if we don't have a name

        return enum_cls
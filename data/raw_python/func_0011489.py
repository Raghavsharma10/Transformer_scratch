def _handle_struct(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_struct.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling struct")

        if node.args is not None:
            for param in node.args.params:
                param.is_func_param = True

        # it's actually being defined
        if node.decls is not None:
            struct_cls = StructUnionDef("struct", self, node)

            if node.name is not None:
                scope.add_type_class(node.name, struct_cls)

            return struct_cls

        # it's declaring a struct field. E.g.
        #    struct IFD subDir;
        else:
            return scope.get_type(node.name)
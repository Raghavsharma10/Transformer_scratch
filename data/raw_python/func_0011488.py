def _handle_struct_call_type_decl(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_struct_call_type_decl.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling struct with parameters")

        struct_cls = self._handle_node(node.type, scope, ctxt, stream)
        struct_args = self._handle_node(node.args, scope, ctxt, stream)

        res = StructDeclWithParams(scope, struct_cls, struct_args)
        return res
def _handle_type_decl(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_type_decl.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling type decl")
        decl = self._handle_node(node.type, scope, ctxt, stream)
        return decl
def _handle_decl_list(self, node, scope, ctxt, stream):
        """Handle For nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling decl list")
        # just handle each declaration
        for decl in node.decls:
            self._handle_node(decl, scope, ctxt, stream)
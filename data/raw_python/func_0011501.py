def _handle_func_def(self, node, scope, ctxt, stream):
        """Handle FuncDef nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling function definition")
        func = self._handle_node(node.decl, scope, ctxt, stream)
        func.body = node.body
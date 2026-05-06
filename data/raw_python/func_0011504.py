def _handle_func_call(self, node, scope, ctxt, stream):
        """Handle FuncCall nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling function call to '{}'".format(node.name.name))
        if node.args is None:
            func_args = []
        else:
            func_args = self._handle_node(node.args, scope, ctxt, stream)
        func = self._handle_node(node.name, scope, ctxt, stream)
        return func.call(func_args, ctxt, scope, stream, self, node.coord)
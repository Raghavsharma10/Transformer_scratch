def _handle_cast(self, node, scope, ctxt, stream):
        """Handle cast nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling cast")
        to_type = self._handle_node(node.to_type, scope, ctxt, stream)
        val_to_cast = self._handle_node(node.expr, scope, ctxt, stream)

        res = to_type()
        res._pfp__set_value(val_to_cast)
        return res
def _handle_if(self, node, scope, ctxt, stream):
        """Handle If nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling if/ternary_op")
        cond = self._handle_node(node.cond, scope, ctxt, stream)
        if cond:
            # there should always be an iftrue
            return self._handle_node(node.iftrue, scope, ctxt, stream)
        else:
            if node.iffalse is not None:
                return self._handle_node(node.iffalse, scope, ctxt, stream)
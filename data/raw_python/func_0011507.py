def _handle_return(self, node, scope, ctxt, stream):
        """Handle Return nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling return")
        if node.expr is None:
            ret_val = None
        else:
            ret_val = self._handle_node(node.expr, scope, ctxt, stream)
        self._dlog("return value = {}".format(ret_val))
        raise errors.InterpReturn(ret_val)
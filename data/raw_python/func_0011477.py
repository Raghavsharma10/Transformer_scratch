def _handle_typename(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_typename

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling typename")

        return self._handle_node(node.type, scope, ctxt, stream)
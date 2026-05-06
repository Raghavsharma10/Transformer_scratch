def _handle_array_ref(self, node, scope, ctxt, stream):
        """Handle ArrayRef nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        ary = self._handle_node(node.name, scope, ctxt, stream)
        subscript = self._handle_node(node.subscript, scope, ctxt, stream)
        return ary[fields.PYVAL(subscript)]
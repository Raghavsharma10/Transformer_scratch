def _handle_exists(self, node, scope, ctxt, stream):
        """Handle the exists unary operator

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        res = fields.Int()
        try:
            self._handle_node(node.expr, scope, ctxt, stream)
            res._pfp__set_value(1)
        except AttributeError:
            res._pfp__set_value(0)
        return res
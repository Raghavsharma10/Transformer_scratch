def _handle_function_exists(self, node, scope, ctxt, stream):
        """Handle the function_exists unary operator

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        res = fields.Int()
        try:
            func = self._handle_node(node.expr, scope, ctxt, stream)
            if isinstance(func, functions.BaseFunction):
                res._pfp__set_value(1)
            else:
                res._pfp__set_value(0)
        except errors.UnresolvedID:
            res._pfp__set_value(0)
        return res
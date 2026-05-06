def _handle_expr_list(self, node, scope, ctxt, stream):
        """Handle ExprList nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling expression list")
        exprs = [
            self._handle_node(expr, scope, ctxt, stream) for expr in node.exprs
        ]
        return exprs
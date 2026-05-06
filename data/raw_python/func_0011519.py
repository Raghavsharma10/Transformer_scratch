def _get_value(self, node, scope, ctxt, stream):
        """Return the value of the node. It is expected to be
        either an AST.ID instance or a constant

        :node: TODO
        :returns: TODO

        """

        res = self._handle_node(node, scope, ctxt, stream)

        if isinstance(res, fields.Field):
            return res._pfp__value

        # assume it's a constant
        else:
            return res
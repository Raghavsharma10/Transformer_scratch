def _handle_parentof(self, node, scope, ctxt, stream):
        """Handle the parentof unary operator

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        # if someone does something like parentof(this).blah,
        # we'll end up with a StructRef instead of an ID ref
        # for node.expr, but we'll also end up with a structref
        # if the user does parentof(a.b.c)...
        # 
        # TODO how to differentiate between the two??
        #
        # the proper way would be to do (parentof(a.b.c)).a or
        # (parentof a.b.c).a

        field = self._handle_node(node.expr, scope, ctxt, stream)
        parent = field._pfp__parent
        return parent
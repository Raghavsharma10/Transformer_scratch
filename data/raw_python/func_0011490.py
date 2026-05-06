def _handle_identifier_type(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_identifier_type.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling identifier")

        cls = self._resolve_to_field_class(node.names, scope)
        return cls
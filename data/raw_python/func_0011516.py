def _handle_continue(self, node, scope, ctxt, stream):
        """Handle continue node

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling continue")
        raise errors.InterpContinue()
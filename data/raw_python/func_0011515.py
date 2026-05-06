def _handle_break(self, node, scope, ctxt, stream):
        """Handle break node

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling break")
        raise errors.InterpBreak()
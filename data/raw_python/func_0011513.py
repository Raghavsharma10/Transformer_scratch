def _handle_while(self, node, scope, ctxt, stream):
        """Handle break node

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling while")
        while node.cond is None or self._handle_node(node.cond, scope, ctxt, stream):
            if node.stmt is not None:
                try:
                    self._handle_node(node.stmt, scope, ctxt, stream)
                except errors.InterpBreak as e:
                    break
                except errors.InterpContinue as e:
                    pass
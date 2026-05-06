def _handle_for(self, node, scope, ctxt, stream):
        """Handle For nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling for")
        if node.init is not None:
            # perform the init
            self._handle_node(node.init, scope, ctxt, stream)

        while node.cond is None or self._handle_node(node.cond, scope, ctxt, stream):
            if node.stmt is not None:
                try:
                    # do the for body
                    self._handle_node(node.stmt, scope, ctxt, stream)
                except errors.InterpBreak as e:
                    break
                
                # we still need to interpret the "next" statement,
                # so just pass
                except errors.InterpContinue as e:
                    pass

            if node.next is not None:
                # do the next statement
                self._handle_node(node.next, scope, ctxt, stream)
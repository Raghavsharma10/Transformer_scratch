def _handle_compound(self, node, scope, ctxt, stream):
        """Handle Compound nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling compound statement")
        #scope.push()

        try:
            for child in node.children():
                self._handle_node(child, scope, ctxt, stream)

        # in case a return occurs, be sure to pop the scope
        # (returns are implemented by raising an exception)
        finally:
            #scope.pop()
            pass
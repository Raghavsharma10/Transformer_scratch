def _handle_init_list(self, node, scope, ctxt, stream):
        """Handle InitList nodes (e.g. when initializing a struct)

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling init list")
        res = []
        for _,init_child in node.children():
            init_field = self._handle_node(init_child, scope, ctxt, stream)
            res.append(init_field)
        return res
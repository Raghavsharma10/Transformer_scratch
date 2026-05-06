def _handle_file_ast(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_file_ast.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._root = ctxt = fields.Dom(stream)
        ctxt._pfp__scope = scope
        self._root._pfp__name = "__root"
        self._root._pfp__interp = self
        self._dlog("handling file AST with {} children".format(len(node.children())))

        for child in node.children():
            self._handle_node(child, scope, ctxt, stream)

        ctxt._pfp__process_fields_metadata()

        return ctxt
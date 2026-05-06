def _handle_func_decl(self, node, scope, ctxt, stream):
        """Handle FuncDecl nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling func decl")

        if node.args is not None:
            # could just call _handle_param_list directly...
            for param in node.args.params:
                # see the check in _handle_decl for how this is kept from
                # being added to the local context/scope
                param.is_func_param = True
            params = self._handle_node(node.args, scope, ctxt, stream)
        else:
            params = functions.ParamListDef([], node.coord)

        func_type = self._handle_node(node.type, scope, ctxt, stream)

        func = functions.Function(func_type, params, scope)

        return func
def _handle_byref_decl(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_byref_decl.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling byref decl")
        field = self._handle_node(node.type.type, scope, ctxt, stream)

        # this will not really be used (maybe except for introspection)
        # with byref function params
        # see issue #35 - we need to wrap the field cls so that the byref
        # doesn't permanently stay on the class
        field = functions.ParamClsWrapper(field)
        field.byref = True
        return field
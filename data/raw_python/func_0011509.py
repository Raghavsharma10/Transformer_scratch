def _handle_array_decl(self, node, scope, ctxt, stream):
        """Handle ArrayDecl nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling array declaration '{}'".format(node.type.declname))

        if node.dim is None:
            # will be used
            array_size = None
        else:
            array_size = self._handle_node(node.dim, scope, ctxt, stream)
        self._dlog("array size = {}".format(array_size))
        # TODO node.dim_quals
        # node.type
        field_cls = self._handle_node(node.type, scope, ctxt, stream)
        self._dlog("field class = {}".format(field_cls))
        array = ArrayDecl(field_cls, array_size)
        #array = fields.Array(array_size, field_cls)
        array._pfp__name = node.type.declname
        #array._pfp__parse(stream)
        return array
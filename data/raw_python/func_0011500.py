def _handle_assignment(self, node, scope, ctxt, stream):
        """Handle assignment nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        def add_op(x,y): x += y
        def sub_op(x,y): x -= y
        def div_op(x,y): x /= y
        def mod_op(x,y): x %= y
        def mul_op(x,y): x *= y
        def xor_op(x,y): x ^= y
        def and_op(x,y): x &= y
        def or_op(x,y): x |= y
        def lshift_op(x,y): x <<= y
        def rshift_op(x,y): x >>= y
        def assign_op(x,y): x._pfp__set_value(y)

        switch = {
            "+="    : add_op,
            "-="    : sub_op,
            "/="    : div_op,
            "%="    : mod_op,
            "*="    : mul_op,
            "^="    : xor_op,
            "&="    : and_op,
            "|="    : or_op,
            "<<="    : lshift_op,
            ">>="    : rshift_op,
            "="        : assign_op
        }

        self._dlog("handling assignment")
        field = self._handle_node(node.lvalue, scope, ctxt, stream)
        self._dlog("field = {}".format(field))
        value = self._handle_node(node.rvalue, scope, ctxt, stream)

        if node.op is None:
            self._dlog("value = {}".format(value))
            field._pfp__set_value(value)
        else:
            self._dlog("value {}= {}".format(node.op, value))
            if node.op not in switch:
                raise errors.UnsupportedAssignmentOperator(node.coord, node.op)
            switch[node.op](field, value)
def _handle_binary_op(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_binary_op.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling binary operation {}".format(node.op))
        switch = {
            "+": lambda x,y: x+y,
            "-": lambda x,y: x-y,
            "*": lambda x,y: x*y,
            "/": lambda x,y: x/y,
            "|": lambda x,y: x|y,
            "^": lambda x,y: x^y,
            "&": lambda x,y: x&y,
            "%": lambda x,y: x%y,
            ">": lambda x,y: x>y,
            "<": lambda x,y: x<y,
            "||": lambda x,y: x or y,
            ">=": lambda x,y: x>=y,
            "<=": lambda x,y: x<=y,
            "==": lambda x,y: x == y,
            "!=": lambda x,y: x != y,
            "&&": lambda x,y: x and y,
            ">>": lambda x,y: x >> y,
            "<<": lambda x,y: x << y,
        }

        left_val = self._handle_node(node.left, scope, ctxt, stream)
        right_val = self._handle_node(node.right, scope, ctxt, stream)

        if node.op not in switch:
            raise errors.UnsupportedBinaryOperator(node.coord, node.op)

        res = switch[node.op](left_val, right_val)

        if type(res) is bool:
            new_res = fields.Int()
            if res:
                new_res._pfp__set_value(1)
            else:
                new_res._pfp__set_value(0)
            res = new_res

        return res
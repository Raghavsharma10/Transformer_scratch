def _handle_unary_op(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_unary_op.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling unary op {}".format(node.op))

        special_switch = {
            "parentof"            : self._handle_parentof,
            "exists"            : self._handle_exists,
            "function_exists"    : self._handle_function_exists,
            "p++"                : self._handle_post_plus_plus,
            "p--"                : self._handle_post_minus_minus,
        }

        switch = {
            # for ++i and --i
            "++":        lambda x,v: x.__iadd__(1),
            "--":        lambda x,v: x.__isub__(1),

            "~":        lambda x,v: ~x,
            "!":        lambda x,v: not x,
            "-":        lambda x,v: -x,
            "sizeof":    lambda x,v: (fields.UInt64()+x._pfp__width()),
            "startof":    lambda x,v: (fields.UInt64()+x._pfp__offset),
        }

        if node.op not in switch and node.op not in special_switch:
            raise errors.UnsupportedUnaryOperator(node.coord, node.op)

        if node.op in special_switch:
            return special_switch[node.op](node, scope, ctxt, stream)

        field = self._handle_node(node.expr, scope, ctxt, stream)
        if type(field) is type:
            field = field()
        res = switch[node.op](field, 1)
        if type(res) is bool:
            new_res = field.__class__()
            new_res._pfp__set_value(1 if res == True else 0)
            res = new_res
        return res
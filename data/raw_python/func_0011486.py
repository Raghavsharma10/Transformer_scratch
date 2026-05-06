def _handle_union(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_union.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling union")

        union_cls = StructUnionDef("union", self, node)
        return union_cls
def _handle_id(self, node, scope, ctxt, stream):
        """Handle an ID node (return a field object for the ID)

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        if node.name == "__root":
            return self._root
        if node.name == "__this" or node.name == "this":
            return ctxt

        self._dlog("handling id {}".format(node.name))
        field = scope.get_id(node.name)

        is_lazy = getattr(node, "is_lazy", False)

        if field is None and not is_lazy:
            raise errors.UnresolvedID(node.coord, node.name)
        elif is_lazy:
            return LazyField(node.name, scope)

        return field
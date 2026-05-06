def _fill_scope_refs(name, scope):
        """Put referenced name in 'ref' dictionary of a scope.

        Walks up the scope tree and adds the name to 'ref' of every scope
        up in the tree until a scope that defines referenced name is reached.
        """
        symbol = scope.resolve(name)
        if symbol is None:
            return

        orig_scope = symbol.scope
        scope.refs[name] = orig_scope
        while scope is not orig_scope:
            scope = scope.get_enclosing_scope()
            scope.refs[name] = orig_scope
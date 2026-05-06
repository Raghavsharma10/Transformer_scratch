def mangle_scope_tree(root, toplevel):
    """Walk over a scope tree and mangle symbol names.

    Args:
        toplevel: Defines if global scope should be mangled or not.
    """
    def mangle(scope):
        # don't mangle global scope if not specified otherwise
        if scope.get_enclosing_scope() is None and not toplevel:
            return
        for name in scope.symbols:
            mangled_name = scope.get_next_mangled_name()
            scope.mangled[name] = mangled_name
            scope.rev_mangled[mangled_name] = name

    def visit(node):
        mangle(node)
        for child in node.children:
            visit(child)

    visit(root)
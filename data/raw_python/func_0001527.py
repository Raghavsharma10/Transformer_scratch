def generic_visit(self, node: ast.AST) -> None:
        """Raise an exception that this node has not been handled."""
        raise NotImplementedError("Unhandled recomputation of the node: {} {}".format(type(node), node))
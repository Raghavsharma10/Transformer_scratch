def visit_Return(self, node: ast.Return) -> Any:  # pylint: disable=no-self-use
        """Raise an exception that this node is unexpected."""
        raise AssertionError("Unexpected return node during the re-computation: {}".format(ast.dump(node)))
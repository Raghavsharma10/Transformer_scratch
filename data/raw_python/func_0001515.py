def visit_IfExp(self, node: ast.IfExp) -> Any:
        """Visit the ``test``, and depending on its outcome, the ``body`` or ``orelse``."""
        test = self.visit(node=node.test)

        if test:
            result = self.visit(node=node.body)
        else:
            result = self.visit(node=node.orelse)

        self.recomputed_values[node] = result
        return result
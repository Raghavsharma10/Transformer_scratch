def visit_Expr(self, node: ast.Expr) -> Any:
        """Visit the node's ``value``."""
        result = self.visit(node=node.value)

        self.recomputed_values[node] = result
        return result
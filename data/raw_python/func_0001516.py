def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Visit the node's ``value`` and get the attribute from the result."""
        value = self.visit(node=node.value)
        if not isinstance(node.ctx, ast.Load):
            raise NotImplementedError(
                "Can only compute a value of Load on the attribute {}, but got context: {}".format(node.attr, node.ctx))

        result = getattr(value, node.attr)

        self.recomputed_values[node] = result
        return result
def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        """Recursively visit the operands and apply the operation on them."""
        values = [self.visit(value_node) for value_node in node.values]

        if isinstance(node.op, ast.And):
            result = functools.reduce(lambda left, right: left and right, values, True)
        elif isinstance(node.op, ast.Or):
            result = functools.reduce(lambda left, right: left or right, values, True)
        else:
            raise NotImplementedError("Unhandled op of {}: {}".format(node, node.op))

        self.recomputed_values[node] = result
        return result
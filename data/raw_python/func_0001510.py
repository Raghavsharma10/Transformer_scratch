def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Visit the node operand and apply the operation on the result."""
        if isinstance(node.op, ast.UAdd):
            result = +self.visit(node=node.operand)
        elif isinstance(node.op, ast.USub):
            result = -self.visit(node=node.operand)
        elif isinstance(node.op, ast.Not):
            result = not self.visit(node=node.operand)
        elif isinstance(node.op, ast.Invert):
            result = ~self.visit(node=node.operand)
        else:
            raise NotImplementedError("Unhandled op of {}: {}".format(node, node.op))

        self.recomputed_values[node] = result
        return result
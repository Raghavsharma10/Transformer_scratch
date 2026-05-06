def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Recursively visit the left and right operand, respectively, and apply the operation on the results."""
        # pylint: disable=too-many-branches
        left = self.visit(node=node.left)
        right = self.visit(node=node.right)

        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            result = left**right
        elif isinstance(node.op, ast.LShift):
            result = left << right
        elif isinstance(node.op, ast.RShift):
            result = left >> right
        elif isinstance(node.op, ast.BitOr):
            result = left | right
        elif isinstance(node.op, ast.BitXor):
            result = left ^ right
        elif isinstance(node.op, ast.BitAnd):
            result = left & right
        elif isinstance(node.op, ast.MatMult):
            result = left @ right
        else:
            raise NotImplementedError("Unhandled op of {}: {}".format(node, node.op))

        self.recomputed_values[node] = result
        return result
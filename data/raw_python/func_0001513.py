def visit_Compare(self, node: ast.Compare) -> Any:
        """Recursively visit the comparators and apply the operations on them."""
        # pylint: disable=too-many-branches
        left = self.visit(node=node.left)

        comparators = [self.visit(node=comparator) for comparator in node.comparators]

        result = None  # type: Optional[Any]
        for comparator, op in zip(comparators, node.ops):
            if isinstance(op, ast.Eq):
                comparison = left == comparator
            elif isinstance(op, ast.NotEq):
                comparison = left != comparator
            elif isinstance(op, ast.Lt):
                comparison = left < comparator
            elif isinstance(op, ast.LtE):
                comparison = left <= comparator
            elif isinstance(op, ast.Gt):
                comparison = left > comparator
            elif isinstance(op, ast.GtE):
                comparison = left >= comparator
            elif isinstance(op, ast.Is):
                comparison = left is comparator
            elif isinstance(op, ast.IsNot):
                comparison = left is not comparator
            elif isinstance(op, ast.In):
                comparison = left in comparator
            elif isinstance(op, ast.NotIn):
                comparison = left not in comparator
            else:
                raise NotImplementedError("Unhandled op of {}: {}".format(node, op))

            if result is None:
                result = comparison
            else:
                result = result and comparison

            left = comparator

        self.recomputed_values[node] = result
        return result
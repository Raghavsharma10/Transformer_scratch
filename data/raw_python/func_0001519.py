def visit_ExtSlice(self, node: ast.ExtSlice) -> Tuple[Any, ...]:
        """Visit each dimension of the advanced slicing and assemble the dimensions in a tuple."""
        result = tuple(self.visit(node=dim) for dim in node.dims)

        self.recomputed_values[node] = result
        return result
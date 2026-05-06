def visit_Dict(self, node: ast.Dict) -> Dict[Any, Any]:
        """Visit keys and values and assemble a dictionary with the results."""
        recomputed_dict = dict()  # type: Dict[Any, Any]
        for key, val in zip(node.keys, node.values):
            recomputed_dict[self.visit(node=key)] = self.visit(node=val)

        self.recomputed_values[node] = recomputed_dict
        return recomputed_dict
def visit_Name(self, node: ast.Name) -> Any:
        """Load the variable by looking it up in the variable look-up and in the built-ins."""
        if not isinstance(node.ctx, ast.Load):
            raise NotImplementedError("Can only compute a value of Load on a name {}, but got context: {}".format(
                node.id, node.ctx))

        result = None  # type: Optional[Any]

        if node.id in self._name_to_value:
            result = self._name_to_value[node.id]

        if result is None and hasattr(builtins, node.id):
            result = getattr(builtins, node.id)

        if result is None and node.id != "None":
            # The variable refers to a name local of the lambda (e.g., a target in the generator expression).
            # Since we evaluate generator expressions with runtime compilation, None is returned here as a placeholder.
            return PLACEHOLDER

        self.recomputed_values[node] = result
        return result
def source_to_code(self, nodes, path, *, _optimize=-1):
        """* Convert the current source to ast 
        * Apply ast transformers.
        * Compile the code."""
        if not isinstance(nodes, ast.Module):
            nodes = ast.parse(nodes, self.path)
        if self._markdown_docstring:
            nodes = update_docstring(nodes)
        return super().source_to_code(
            ast.fix_missing_locations(self.visit(nodes)), path, _optimize=_optimize
        )
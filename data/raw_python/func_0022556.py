def _guess_name_of(self, expr):
        """Tries to guess what variable name 'expr' ends in.

        This is a heuristic that roughly emulates what most SQL databases
        name columns, based on selected variable names or applied functions.
        """
        if isinstance(expr, ast.Var):
            return expr.value

        if isinstance(expr, ast.Resolve):
            # We know the RHS of resolve is a Literal because that's what
            # Parser.dot_rhs does.
            return expr.rhs.value

        if isinstance(expr, ast.Select) and isinstance(expr.rhs, ast.Literal):
            name = self._guess_name_of(expr.lhs)
            if name is not None:
                return "%s_%s" % (name, expr.rhs.value)

        if isinstance(expr, ast.Apply) and isinstance(expr.func, ast.Var):
            return expr.func.value
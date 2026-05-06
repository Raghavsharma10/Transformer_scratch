def eval(self, expr):
        """
        Evaluates an expression

        :param expr: Expression to evaluate
        :return: Result of expression
        """
        # set a copy of the expression aside, so we can give nice errors...

        self.expr = expr

        # and evaluate:
        return self._eval(ast.parse(expr.strip()).body[0].value)
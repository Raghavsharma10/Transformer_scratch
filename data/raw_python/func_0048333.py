def p_conditional_expr(self, p):
        """
        conditional_expr \
            : logical_or_expr
            | logical_or_expr CONDOP assignment_expr COLON assignment_expr
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.Conditional(
                predicate=p[1], consequent=p[3], alternative=p[5])
def p_expr_noin(self, p):
        """expr_noin : assignment_expr_noin
                     | expr_noin COMMA assignment_expr_noin
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.Comma(left=p[1], right=p[3])
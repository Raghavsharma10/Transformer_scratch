def p_expr_nobf(self, p):
        """expr_nobf : assignment_expr_nobf
                     | expr_nobf COMMA assignment_expr
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.Comma(left=p[1], right=p[3])
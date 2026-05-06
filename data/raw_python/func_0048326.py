def p_member_expr_nobf(self, p):
        """member_expr_nobf : primary_expr_no_brace
                            | function_expr
                            | member_expr_nobf LBRACKET expr RBRACKET
                            | member_expr_nobf PERIOD identifier
                            | NEW member_expr arguments
        """
        if len(p) == 2:
            p[0] = p[1]
        elif p[1] == 'new':
            p[0] = ast.NewExpr(p[2], p[3])
        elif p[2] == '.':
            p[0] = ast.DotAccessor(p[1], p[3])
        else:
            p[0] = ast.BracketAccessor(p[1], p[3])
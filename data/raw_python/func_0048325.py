def p_member_expr(self, p):
        """member_expr : primary_expr
                       | function_expr
                       | member_expr LBRACKET expr RBRACKET
                       | member_expr PERIOD identifier
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
def p_call_expr(self, p):
        """call_expr : member_expr arguments
                     | call_expr arguments
                     | call_expr LBRACKET expr RBRACKET
                     | call_expr PERIOD identifier
        """
        if len(p) == 3:
            p[0] = ast.FunctionCall(p[1], p[2])
        elif len(p) == 4:
            p[0] = ast.DotAccessor(p[1], p[3])
        else:
            p[0] = ast.BracketAccessor(p[1], p[3])
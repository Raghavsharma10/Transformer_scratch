def p_call_expr_nobf(self, p):
        """call_expr_nobf : member_expr_nobf arguments
                          | call_expr_nobf arguments
                          | call_expr_nobf LBRACKET expr RBRACKET
                          | call_expr_nobf PERIOD identifier
        """
        if len(p) == 3:
            p[0] = ast.FunctionCall(p[1], p[2])
        elif len(p) == 4:
            p[0] = ast.DotAccessor(p[1], p[3])
        else:
            p[0] = ast.BracketAccessor(p[1], p[3])
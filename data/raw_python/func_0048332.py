def p_multiplicative_expr(self, p):
        """multiplicative_expr : unary_expr
                               | multiplicative_expr MULT unary_expr
                               | multiplicative_expr DIV unary_expr
                               | multiplicative_expr MOD unary_expr
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.BinOp(op=p[2], left=p[1], right=p[3])
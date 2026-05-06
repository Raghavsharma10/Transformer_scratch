def p_unary_expr_common(self, p):
        """unary_expr_common : DELETE unary_expr
                             | VOID unary_expr
                             | TYPEOF unary_expr
                             | PLUSPLUS unary_expr
                             | MINUSMINUS unary_expr
                             | PLUS unary_expr
                             | MINUS unary_expr
                             | BNOT unary_expr
                             | NOT unary_expr
        """
        p[0] = self.asttypes.UnaryExpr(p[1], p[2])
        p[0].setpos(p)
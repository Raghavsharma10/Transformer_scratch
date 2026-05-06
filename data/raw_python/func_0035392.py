def p_new_expr_nobf(self, p):
        """new_expr_nobf : member_expr_nobf
                         | NEW new_expr
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = self.asttypes.NewExpr(p[2])
            p[0].setpos(p)
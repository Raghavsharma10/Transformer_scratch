def p_primary_expr_no_brace_4(self, p):
        """primary_expr_no_brace : LPAREN expr RPAREN"""
        if isinstance(p[2], self.asttypes.GroupingOp):
            # this reduces the grouping operator to one.
            p[0] = p[2]
        else:
            p[0] = self.asttypes.GroupingOp(expr=p[2])
            p[0].setpos(p)
def p_primary_expr_no_brace_2(self, p):
        """primary_expr_no_brace : THIS"""
        p[0] = self.asttypes.This()
        p[0].setpos(p)
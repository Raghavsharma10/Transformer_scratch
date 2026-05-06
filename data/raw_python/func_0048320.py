def p_primary_expr_no_brace_1(self, p):
        """primary_expr_no_brace : identifier"""
        p[1]._mangle_candidate = True
        p[1]._in_expression = True
        p[0] = p[1]
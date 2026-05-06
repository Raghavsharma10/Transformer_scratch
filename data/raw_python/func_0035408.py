def p_expr_statement(self, p):
        """expr_statement : expr_nobf SEMI
                          | expr_nobf AUTOSEMI
        """
        # In 12.4, expression statements cannot start with either the
        # 'function' keyword or '{'.  However, the lexing and production
        # of the FuncExpr nodes can be done through further rules have
        # been done, so flag this as an exception, but must be raised
        # like so due to avoid the SyntaxError being flagged by ply and
        # which would result in an infinite loop in this case.

        if isinstance(p[1], self.asttypes.FuncExpr):
            _, line, col = p[1].getpos('(', 0)
            raise ProductionError(ECMASyntaxError(
                'Function statement requires a name at %s:%s' % (line, col)))

        # The most bare 'block' rule is defined as part of 'statement'
        # and there are no other bare rules that would result in the
        # production of such like for 'function_expr'.

        p[0] = self.asttypes.ExprStatement(p[1])
        p[0].setpos(p)
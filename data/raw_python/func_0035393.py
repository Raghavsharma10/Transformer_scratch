def p_call_expr(self, p):
        """call_expr : member_expr arguments
                     | call_expr arguments
                     | call_expr LBRACKET expr RBRACKET
                     | call_expr PERIOD identifier_name_string
        """
        if len(p) == 3:
            p[0] = self.asttypes.FunctionCall(p[1], p[2])
            p[0].setpos(p)  # require yacc_tracking
        elif len(p) == 4:
            p[0] = self.asttypes.DotAccessor(p[1], p[3])
            p[0].setpos(p, 2)
        else:
            p[0] = self.asttypes.BracketAccessor(p[1], p[3])
            p[0].setpos(p, 2)
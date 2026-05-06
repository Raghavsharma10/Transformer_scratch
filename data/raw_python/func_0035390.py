def p_member_expr(self, p):
        """member_expr : primary_expr
                       | function_expr
                       | member_expr LBRACKET expr RBRACKET
                       | member_expr PERIOD identifier_name_string
                       | NEW member_expr arguments
        """
        if len(p) == 2:
            p[0] = p[1]
            return

        if p[1] == 'new':
            p[0] = self.asttypes.NewExpr(p[2], p[3])
            p[0].setpos(p)
        elif p[2] == '.':
            p[0] = self.asttypes.DotAccessor(p[1], p[3])
            p[0].setpos(p, 2)
        else:
            p[0] = self.asttypes.BracketAccessor(p[1], p[3])
            p[0].setpos(p, 2)
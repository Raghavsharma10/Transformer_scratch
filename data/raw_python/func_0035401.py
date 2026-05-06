def p_expr(self, p):
        """expr : assignment_expr
                | expr COMMA assignment_expr
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = self.asttypes.Comma(left=p[1], right=p[3])
            p[0].setpos(p, 2)
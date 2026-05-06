def p_null_literal(self, p):
        """null_literal : NULL"""
        p[0] = self.asttypes.Null(p[1])
        p[0].setpos(p)
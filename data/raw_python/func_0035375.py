def p_numeric_literal(self, p):
        """numeric_literal : NUMBER"""
        p[0] = self.asttypes.Number(p[1])
        p[0].setpos(p)
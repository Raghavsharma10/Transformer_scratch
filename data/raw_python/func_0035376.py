def p_string_literal(self, p):
        """string_literal : STRING"""
        p[0] = self.asttypes.String(p[1])
        p[0].setpos(p)
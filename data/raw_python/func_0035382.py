def p_array_literal_1(self, p):
        """array_literal : LBRACKET elision_opt RBRACKET"""
        p[0] = self.asttypes.Array(items=p[2])
        p[0].setpos(p)
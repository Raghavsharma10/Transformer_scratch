def p_identifier(self, p):
        """identifier : ID"""
        p[0] = self.asttypes.Identifier(p[1])
        p[0].setpos(p)
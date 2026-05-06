def p_program(self, p):
        """program : source_elements"""
        p[0] = self.asttypes.ES5Program(p[1])
        p[0].setpos(p)
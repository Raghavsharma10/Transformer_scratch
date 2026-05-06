def p_throw_statement(self, p):
        """throw_statement : THROW expr SEMI
                           | THROW expr AUTOSEMI
        """
        p[0] = self.asttypes.Throw(expr=p[2])
        p[0].setpos(p)
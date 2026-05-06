def p_break_statement_1(self, p):
        """break_statement : BREAK SEMI
                           | BREAK AUTOSEMI
        """
        p[0] = self.asttypes.Break()
        p[0].setpos(p)
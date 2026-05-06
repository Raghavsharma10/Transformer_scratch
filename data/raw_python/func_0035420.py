def p_break_statement_2(self, p):
        """break_statement : BREAK identifier SEMI
                           | BREAK identifier AUTOSEMI
        """
        p[0] = self.asttypes.Break(p[2])
        p[0].setpos(p)
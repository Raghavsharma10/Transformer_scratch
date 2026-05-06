def p_continue_statement_2(self, p):
        """continue_statement : CONTINUE identifier SEMI
                              | CONTINUE identifier AUTOSEMI
        """
        p[0] = self.asttypes.Continue(p[2])
        p[0].setpos(p)
def p_continue_statement_1(self, p):
        """continue_statement : CONTINUE SEMI
                              | CONTINUE AUTOSEMI
        """
        p[0] = self.asttypes.Continue()
        p[0].setpos(p)
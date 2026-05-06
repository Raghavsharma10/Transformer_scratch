def p_return_statement_1(self, p):
        """return_statement : RETURN SEMI
                            | RETURN AUTOSEMI
        """
        p[0] = self.asttypes.Return()
        p[0].setpos(p)
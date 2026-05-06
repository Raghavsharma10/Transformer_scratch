def p_return_statement_2(self, p):
        """return_statement : RETURN expr SEMI
                            | RETURN expr AUTOSEMI
        """
        p[0] = self.asttypes.Return(expr=p[2])
        p[0].setpos(p)
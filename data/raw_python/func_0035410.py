def p_if_statement_2(self, p):
        """if_statement : IF LPAREN expr RPAREN statement ELSE statement"""
        p[0] = self.asttypes.If(
            predicate=p[3], consequent=p[5], alternative=p[7])
        p[0].setpos(p)
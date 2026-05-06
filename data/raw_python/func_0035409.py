def p_if_statement_1(self, p):
        """if_statement : IF LPAREN expr RPAREN statement"""
        p[0] = self.asttypes.If(predicate=p[3], consequent=p[5])
        p[0].setpos(p)
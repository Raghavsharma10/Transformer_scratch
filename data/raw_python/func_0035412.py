def p_iteration_statement_2(self, p):
        """iteration_statement : WHILE LPAREN expr RPAREN statement"""
        p[0] = self.asttypes.While(predicate=p[3], statement=p[5])
        p[0].setpos(p)
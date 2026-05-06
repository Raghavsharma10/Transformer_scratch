def p_iteration_statement_1(self, p):
        """
        iteration_statement \
            : DO statement WHILE LPAREN expr RPAREN SEMI
            | DO statement WHILE LPAREN expr RPAREN AUTOSEMI
        """
        p[0] = self.asttypes.DoWhile(predicate=p[5], statement=p[2])
        p[0].setpos(p)
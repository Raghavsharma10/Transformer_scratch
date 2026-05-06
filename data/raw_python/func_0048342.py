def p_iteration_statement_1(self, p):
        """
        iteration_statement \
            : DO statement WHILE LPAREN expr RPAREN SEMI
            | DO statement WHILE LPAREN expr RPAREN auto_semi
        """
        p[0] = ast.DoWhile(predicate=p[5], statement=p[2])
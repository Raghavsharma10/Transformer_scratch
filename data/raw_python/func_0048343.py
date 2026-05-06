def p_iteration_statement_2(self, p):
        """iteration_statement : WHILE LPAREN expr RPAREN statement"""
        p[0] = ast.While(predicate=p[3], statement=p[5])
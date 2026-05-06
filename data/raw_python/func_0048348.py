def p_with_statement(self, p):
        """with_statement : WITH LPAREN expr RPAREN statement"""
        p[0] = ast.With(expr=p[3], statement=p[5])
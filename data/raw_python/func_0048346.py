def p_iteration_statement_5(self, p):
        """
        iteration_statement : \
            FOR LPAREN VAR identifier IN expr RPAREN statement
        """
        p[0] = ast.ForIn(item=ast.VarDecl(p[4]), iterable=p[6], statement=p[8])
def p_iteration_statement_6(self, p):
        """
        iteration_statement \
          : FOR LPAREN VAR identifier initializer_noin IN expr RPAREN statement
        """
        p[0] = ast.ForIn(item=ast.VarDecl(identifier=p[4], initializer=p[5]),
                         iterable=p[7], statement=p[9])
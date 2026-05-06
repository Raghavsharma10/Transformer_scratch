def p_iteration_statement_5(self, p):
        """
        iteration_statement : \
            FOR LPAREN VAR identifier IN expr RPAREN statement
        """
        vardecl = self.asttypes.VarDeclNoIn(identifier=p[4])
        vardecl.setpos(p, 3)
        p[0] = self.asttypes.ForIn(item=vardecl, iterable=p[6], statement=p[8])
        p[0].setpos(p)
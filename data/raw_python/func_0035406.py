def p_variable_declaration_noin(self, p):
        """variable_declaration_noin : identifier
                                     | identifier initializer_noin
        """
        if len(p) == 2:
            p[0] = self.asttypes.VarDecl(p[1])
            p[0].setpos(p)  # require yacc_tracking
        else:
            p[0] = self.asttypes.VarDecl(p[1], p[2])
            p[0].setpos(p, additional=(('=', 2),))
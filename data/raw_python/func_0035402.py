def p_variable_statement(self, p):
        """variable_statement : VAR variable_declaration_list SEMI
                              | VAR variable_declaration_list AUTOSEMI
        """
        p[0] = self.asttypes.VarStatement(p[2])
        p[0].setpos(p)
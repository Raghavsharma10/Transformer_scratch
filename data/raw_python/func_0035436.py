def p_debugger_statement(self, p):
        """debugger_statement : DEBUGGER SEMI
                              | DEBUGGER AUTOSEMI
        """
        p[0] = self.asttypes.Debugger(p[1])
        p[0].setpos(p)
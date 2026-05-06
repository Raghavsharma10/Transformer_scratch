def p_arguments(self, p):
        """arguments : LPAREN RPAREN
                     | LPAREN argument_list RPAREN
        """
        if len(p) == 4:
            p[0] = self.asttypes.Arguments(p[2])
        else:
            p[0] = self.asttypes.Arguments([])
        p[0].setpos(p)
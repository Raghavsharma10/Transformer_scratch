def p_catch(self, p):
        """catch : CATCH LPAREN identifier RPAREN block"""
        p[0] = self.asttypes.Catch(identifier=p[3], elements=p[5])
        p[0].setpos(p)
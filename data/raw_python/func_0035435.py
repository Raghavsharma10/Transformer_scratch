def p_finally(self, p):
        """finally : FINALLY block"""
        p[0] = self.asttypes.Finally(elements=p[2])
        p[0].setpos(p)
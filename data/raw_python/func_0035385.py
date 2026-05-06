def p_elision(self, p):
        """elision : COMMA
                   | elision COMMA
        """
        if len(p) == 2:
            p[0] = [self.asttypes.Elision(1)]
            p[0][0].setpos(p)
        else:
            # increment the Elision value.
            p[1][-1].value += 1
            p[0] = p[1]
        # TODO there should be a cleaner API for the lexer and their
        # token types for ensuring that the mappings are available.
        p[0][0]._token_map = {(',' * p[0][0].value): [
            p[0][0].findpos(p, 0)]}
        return
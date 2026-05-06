def p_elision(self, p):
        """elision : COMMA
                   | elision COMMA
        """
        if len(p) == 2:
            p[0] = [ast.Elision(p[1])]
        else:
            p[1].append(ast.Elision(p[2]))
            p[0] = p[1]
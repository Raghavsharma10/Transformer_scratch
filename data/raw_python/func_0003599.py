def p_statements(self, p):
        """statements : statements statement
                      | statement
        """
        n = len(p)
        if n == 3:
            p[0] = p[1] + [p[2]]
        elif n == 2:
            p[0] = ['statements', p[1]]
def p_config(self, p):
        """config : config contents
                  | contents
        """
        n = len(p)
        if n == 3:
            p[0] = p[1] + [p[2]]
        elif n == 2:
            p[0] = ['config', p[1]]
def p_formal_parameter_list(self, p):
        """formal_parameter_list : identifier
                                 | formal_parameter_list COMMA identifier
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[1].append(p[3])
            p[0] = p[1]
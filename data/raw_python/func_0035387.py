def p_property_list(self, p):
        """property_list : property_assignment
                         | property_list COMMA property_assignment
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[1].append(p[3])
            p[0] = p[1]
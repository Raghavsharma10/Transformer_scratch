def p_element_list(self, p):
        """element_list : elision_opt assignment_expr
                        | element_list COMMA elision_opt assignment_expr
        """
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[1].extend(p[3])
            p[1].append(p[4])
            p[0] = p[1]
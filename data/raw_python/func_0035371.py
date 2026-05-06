def p_source_element_list(self, p):
        """source_element_list : source_element
                               | source_element_list source_element
        """
        if len(p) == 2:  # single source element
            p[0] = [p[1]]
        else:
            p[1].append(p[2])
            p[0] = p[1]
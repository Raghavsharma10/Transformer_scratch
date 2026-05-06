def p_variable_declaration_list_noin(self, p):
        """
        variable_declaration_list_noin \
            : variable_declaration_noin
            | variable_declaration_list_noin COMMA variable_declaration_noin
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[1].append(p[3])
            p[0] = p[1]
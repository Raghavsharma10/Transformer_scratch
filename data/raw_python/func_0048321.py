def p_array_literal_2(self, p):
        """array_literal : LBRACKET element_list RBRACKET
                         | LBRACKET element_list COMMA elision_opt RBRACKET
        """
        items = p[2]
        if len(p) == 6:
            items.extend(p[4])
        p[0] = ast.Array(items=items)
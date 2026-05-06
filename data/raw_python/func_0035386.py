def p_object_literal(self, p):
        """object_literal : LBRACE RBRACE
                          | LBRACE property_list RBRACE
                          | LBRACE property_list COMMA RBRACE
        """
        if len(p) == 3:
            p[0] = self.asttypes.Object()
        else:
            p[0] = self.asttypes.Object(properties=p[2])
        p[0].setpos(p)
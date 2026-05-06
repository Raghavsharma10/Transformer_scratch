def p_property_assignment(self, p):
        """property_assignment \
             : property_name COLON assignment_expr
             | GETPROP property_name LPAREN RPAREN LBRACE function_body RBRACE
             | SETPROP property_name LPAREN property_set_parameter_list RPAREN\
                   LBRACE function_body RBRACE
        """
        if len(p) == 4:
            p[0] = self.asttypes.Assign(left=p[1], op=p[2], right=p[3])
            p[0].setpos(p, 2)
        elif len(p) == 8:
            p[0] = self.asttypes.GetPropAssign(prop_name=p[2], elements=p[6])
            p[0].setpos(p)
        else:
            p[0] = self.asttypes.SetPropAssign(
                prop_name=p[2], parameter=p[4], elements=p[7])
            p[0].setpos(p)
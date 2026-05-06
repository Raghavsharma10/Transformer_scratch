def p_obj_fields(self, p):
        """
        obj_fields : obj_fields ',' obj_field
                   | obj_field
        """
        p[0] = dict([p[1]] if len(p) == 2 else p[1] + [p[3]])
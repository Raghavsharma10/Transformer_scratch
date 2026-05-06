def p_object_ty(self, p):
        """
        object_ty : OBJECT '(' ID ')'
                  | OBJECT '(' ID ',' obj_fields ')'
        """
        field_types = {} if len(p) == 5 else p[5]
        p[0] = Object(p[3], **field_types)
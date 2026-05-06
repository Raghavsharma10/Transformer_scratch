def p_dynamic_class_name_reference(p):
    '''dynamic_class_name_reference : base_variable OBJECT_OPERATOR object_property dynamic_class_name_variable_properties
                                    | base_variable'''
    if len(p) == 5:
        name, dims = p[3]
        p[0] = ast.ObjectProperty(p[1], name, lineno=p.lineno(2))
        for class_, dim, lineno in dims:
            p[0] = class_(p[0], dim, lineno=lineno)
        for name, dims in p[4]:
            p[0] = ast.ObjectProperty(p[0], name, lineno=p.lineno(2))
            for class_, dim, lineno in dims:
                p[0] = class_(p[0], dim, lineno=lineno)
    else:
        p[0] = p[1]
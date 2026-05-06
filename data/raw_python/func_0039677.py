def p_variable(p):
    '''variable : base_variable_with_function_calls OBJECT_OPERATOR object_property method_or_not variable_properties
                | base_variable_with_function_calls'''
    if len(p) == 6:
        name, dims = p[3]
        params = p[4]
        if params is not None:
            p[0] = ast.MethodCall(p[1], name, params, lineno=p.lineno(2))
        else:
            p[0] = ast.ObjectProperty(p[1], name, lineno=p.lineno(2))
        for class_, dim, lineno in dims:
            p[0] = class_(p[0], dim, lineno=lineno)
        for (name, dims), params in p[5]:
            if params is not None:
                p[0] = ast.MethodCall(p[0], name, params, lineno=p.lineno(2))
            else:
                p[0] = ast.ObjectProperty(p[0], name, lineno=p.lineno(2))
            for class_, dim, lineno in dims:
                p[0] = class_(p[0], dim, lineno=lineno)
    else:
        p[0] = p[1]
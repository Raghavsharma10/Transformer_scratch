def p_static_member(p):
    '''static_member : class_name DOUBLE_COLON variable_without_objects
                     | variable_class_name DOUBLE_COLON variable_without_objects'''
    p[0] = ast.StaticProperty(p[1], p[3], lineno=p.lineno(2))
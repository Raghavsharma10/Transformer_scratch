def p_expr_new(p):
    'expr : NEW class_name_reference ctor_arguments'
    p[0] = ast.New(p[2], p[3], lineno=p.lineno(1))
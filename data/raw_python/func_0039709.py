def p_common_scalar_magic_method(p):
    'common_scalar : METHOD_C'
    p[0] = ast.MagicConstant(p[1].upper(), None, lineno=p.lineno(1))
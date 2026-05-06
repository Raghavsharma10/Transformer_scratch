def p_common_scalar_magic_func(p):
    'common_scalar : FUNC_C'
    p[0] = ast.MagicConstant(p[1].upper(), None, lineno=p.lineno(1))
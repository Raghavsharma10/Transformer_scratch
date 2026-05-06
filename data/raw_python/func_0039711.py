def p_common_scalar_magic_ns(p):
    'common_scalar : NS_C'
    p[0] = ast.MagicConstant(p[1].upper(), None, lineno=p.lineno(1))
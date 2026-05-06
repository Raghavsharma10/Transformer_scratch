def p_common_scalar_magic_class(p):
    'common_scalar : CLASS_C'
    p[0] = ast.MagicConstant(p[1].upper(), None, lineno=p.lineno(1))
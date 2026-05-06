def p_common_scalar_magic_line(p):
    'common_scalar : LINE'
    p[0] = ast.MagicConstant(p[1].upper(), p.lineno(1), lineno=p.lineno(1))
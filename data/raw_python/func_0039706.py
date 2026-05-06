def p_common_scalar_magic_file(p):
    'common_scalar : FILE'
    value = getattr(p.lexer, 'filename', None)
    p[0] = ast.MagicConstant(p[1].upper(), value, lineno=p.lineno(1))
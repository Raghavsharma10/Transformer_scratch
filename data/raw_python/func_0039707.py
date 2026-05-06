def p_common_scalar_magic_dir(p):
    'common_scalar : DIR'
    value = getattr(p.lexer, 'filename', None)
    if value is not None:
        value = os.path.dirname(value)
    p[0] = ast.MagicConstant(p[1].upper(), value, lineno=p.lineno(1))
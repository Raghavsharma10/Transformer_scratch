def p_constant_declaration(p):
    'constant_declaration : STRING EQUALS static_scalar'
    p[0] = ast.ConstantDeclaration(p[1], p[3], lineno=p.lineno(1))
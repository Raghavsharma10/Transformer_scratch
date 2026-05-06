def p_statement_foreach(p):
    'statement : FOREACH LPAREN expr AS foreach_variable foreach_optional_arg RPAREN foreach_statement'
    if p[6] is None:
        p[0] = ast.Foreach(p[3], None, p[5], p[8], lineno=p.lineno(1))
    else:
        p[0] = ast.Foreach(p[3], p[5], p[6], p[8], lineno=p.lineno(1))
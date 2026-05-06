def t_php_OBJECT_OPERATOR(t):
    r'->'
    if re.match(r'[A-Za-z_]', peek(t.lexer)):
        t.lexer.push_state('property')
    return t
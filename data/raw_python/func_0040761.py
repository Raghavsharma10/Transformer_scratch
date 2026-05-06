def t_quoted_DOLLAR_OPEN_CURLY_BRACES(t):
    r'\$\{'
    if re.match(r'[A-Za-z_]', peek(t.lexer)):
        t.lexer.push_state('varname')
    else:
        t.lexer.push_state('php')
    return t
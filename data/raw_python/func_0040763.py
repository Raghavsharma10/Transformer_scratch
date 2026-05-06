def t_quotedvar_DOLLAR_OPEN_CURLY_BRACES(t):
    r'\$\{'
    if re.match(r'[A-Za-z_]', peek(t.lexer)):
        t.lexer.begin('varname')
    else:
        t.lexer.begin('php')
    return t
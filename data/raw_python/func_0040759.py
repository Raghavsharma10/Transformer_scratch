def t_php_CLOSE_TAG(t):
    r'[?%]>\r?\n?'
    t.lexer.lineno += t.value.count("\n")
    t.lexer.begin('INITIAL')
    return t
def t_OPEN_TAG(t):
    r'<[?%]((php[ \t\r\n]?)|=)?'
    if '=' in t.value: t.type = 'OPEN_TAG_WITH_ECHO'
    t.lexer.lineno += t.value.count("\n")
    t.lexer.begin('php')
    return t
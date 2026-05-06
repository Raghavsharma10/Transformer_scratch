def t_quotedvar_ENCAPSED_AND_WHITESPACE(t):
    r'( [^"\\${] | \\(.|\n) | \$(?![A-Za-z_{]) | \{(?!\$) )+'
    t.lexer.lineno += t.value.count("\n")
    t.lexer.pop_state()
    return t
def t_heredocvar_ENCAPSED_AND_WHITESPACE(t):
    r'( [^\n\\${] | \\. | \$(?![A-Za-z_{]) | \{(?!\$) )+\n? | \\?\n'
    t.lexer.lineno += t.value.count("\n")
    t.lexer.pop_state()
    return t
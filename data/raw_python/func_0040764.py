def t_php_START_HEREDOC(t):
    r'<<<[ \t]*(?P<label>[A-Za-z_][\w_]*)\n'
    t.lexer.lineno += t.value.count("\n")
    t.lexer.push_state('heredoc')
    t.lexer.heredoc_label = t.lexer.lexmatch.group('label')
    return t
def t_heredoc_END_HEREDOC(t):
    r'(?<=\n)[A-Za-z_][\w_]*'
    if t.value == t.lexer.heredoc_label:
        del t.lexer.heredoc_label
        t.lexer.pop_state()
    else:
        t.type = 'ENCAPSED_AND_WHITESPACE'
    return t
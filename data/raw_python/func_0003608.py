def t_heredoc_OPTION_AND_VALUE(self, t):
        r'[^\r\n]+'
        if t.value.lstrip() != t.lexer.heredoc_anchor:
            return

        t.type = "OPTION_AND_VALUE"
        t.lexer.begin('INITIAL')

        value = t.lexer.lexdata[t.lexer.code_start + 1:t.lexer.lexpos - len(t.lexer.heredoc_anchor)]

        t.lexer.lineno += len(re.findall(r'\r\n|\n|\r', t.value))

        t.value = t.lexer.heredoc_option, value

        return t
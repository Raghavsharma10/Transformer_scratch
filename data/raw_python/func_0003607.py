def t_multiline_NEWLINE(self, t):
        r'\r\n|\n|\r'
        if t.lexer.multiline_newline_seen:
            return self.t_multiline_OPTION_AND_VALUE(t)
        t.lexer.multiline_newline_seen = True
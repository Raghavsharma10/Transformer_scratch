def t_ccomment_close(self, t):
        r'\*\/'
        t.lexer.ccomment_level -= 1

        if t.lexer.ccomment_level == 0:
            t.value = t.lexer.lexdata[t.lexer.code_start:t.lexer.lexpos + 1 - 3]
            t.type = "CCOMMENT"
            t.lexer.lineno += t.value.count('\n')
            t.lexer.begin('INITIAL')
            return t
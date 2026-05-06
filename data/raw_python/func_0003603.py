def t_CCOMMENT(self, t):
        r'\/\*'
        t.lexer.code_start = t.lexer.lexpos
        t.lexer.ccomment_level = 1  # Initial comment level
        t.lexer.begin('ccomment')
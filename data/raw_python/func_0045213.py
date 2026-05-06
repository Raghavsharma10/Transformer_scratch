def t_ID(self, t):
        r'~?[a-zA-Z_][a-zA-Z0-9_]*'

        if t.value[0] == '~':
            t.type = 'TYVAR'
            t.value = t.value[1:]
        elif t.value in self.reserved_words:
            t.type = self.reserved_words[t.value]
        else:
            t.type = 'ID'

        return t
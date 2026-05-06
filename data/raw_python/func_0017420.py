def t_ID(self, t):
        r'`[^`]*`|[a-zA-Z_][a-zA-Z_0-9:@]*'
        res = self.oper.get(t.value, None)  # Check for reserved words
        if res is None:
            res = t.value.upper()
            if res == 'FALSE':
                t.type = 'BOOL'
                t.value = False
            elif res == 'TRUE':
                t.type = 'BOOL'
                t.value = True
            else:
                t.type = 'ID'
        else:
            t.value = res
            t.type = 'FUNCTION'
        return t
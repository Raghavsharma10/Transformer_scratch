def t_php_STRING(t):
    r'[A-Za-z_][\w_]*'
    t.type = reserved_map.get(t.value.upper(), 'STRING')
    return t
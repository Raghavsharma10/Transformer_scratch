def p_class_declaration_statement(p):
    '''class_declaration_statement : class_entry_type STRING extends_from implements_list LBRACE class_statement_list RBRACE
                                   | INTERFACE STRING interface_extends_list LBRACE class_statement_list RBRACE'''
    if len(p) == 8:
        p[0] = ast.Class(p[2], p[1], p[3], p[4], p[6], lineno=p.lineno(2))
    else:
        p[0] = ast.Interface(p[2], p[3], p[5], lineno=p.lineno(1))
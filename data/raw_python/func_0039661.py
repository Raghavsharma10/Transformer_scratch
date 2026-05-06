def p_case_list(p):
    '''case_list : empty
                 | case_list CASE expr case_separator inner_statement_list
                 | case_list DEFAULT case_separator inner_statement_list'''
    if len(p) == 6:
        p[0] = p[1] + [ast.Case(p[3], p[5], lineno=p.lineno(2))]
    elif len(p) == 5:
        p[0] = p[1] + [ast.Default(p[4], lineno=p.lineno(2))]
    else:
        p[0] = []
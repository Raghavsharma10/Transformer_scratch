def p_function_call(p):
    '''function_call : namespace_name LPAREN function_call_parameter_list RPAREN
                     | NS_SEPARATOR namespace_name LPAREN function_call_parameter_list RPAREN
                     | NAMESPACE NS_SEPARATOR namespace_name LPAREN function_call_parameter_list RPAREN'''
    if len(p) == 5:
        p[0] = ast.FunctionCall(p[1], p[3], lineno=p.lineno(2))
    elif len(p) == 6:
        p[0] = ast.FunctionCall(p[1] + p[2], p[4], lineno=p.lineno(1))
    else:
        p[0] = ast.FunctionCall(p[1] + p[2] + p[3], p[5], lineno=p.lineno(1))
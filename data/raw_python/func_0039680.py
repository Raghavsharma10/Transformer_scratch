def p_function_call_variable(p):
    'function_call : variable_without_objects LPAREN function_call_parameter_list RPAREN'
    p[0] = ast.FunctionCall(p[1], p[3], lineno=p.lineno(2))
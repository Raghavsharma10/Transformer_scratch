def p_function_call_static(p):
    '''function_call : class_name DOUBLE_COLON STRING LPAREN function_call_parameter_list RPAREN
                     | class_name DOUBLE_COLON variable_without_objects LPAREN function_call_parameter_list RPAREN
                     | variable_class_name DOUBLE_COLON STRING LPAREN function_call_parameter_list RPAREN
                     | variable_class_name DOUBLE_COLON variable_without_objects LPAREN function_call_parameter_list RPAREN'''
    p[0] = ast.StaticMethodCall(p[1], p[3], p[5], lineno=p.lineno(2))
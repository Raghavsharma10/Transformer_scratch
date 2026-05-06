def p_expr_binary_op(p):
    '''expr : expr BOOLEAN_AND expr
            | expr BOOLEAN_OR expr
            | expr LOGICAL_AND expr
            | expr LOGICAL_OR expr
            | expr LOGICAL_XOR expr
            | expr AND expr
            | expr OR expr
            | expr XOR expr
            | expr CONCAT expr
            | expr PLUS expr
            | expr MINUS expr
            | expr MUL expr
            | expr DIV expr
            | expr SL expr
            | expr SR expr
            | expr MOD expr
            | expr IS_IDENTICAL expr
            | expr IS_NOT_IDENTICAL expr
            | expr IS_EQUAL expr
            | expr IS_NOT_EQUAL expr
            | expr IS_SMALLER expr
            | expr IS_SMALLER_OR_EQUAL expr
            | expr IS_GREATER expr
            | expr IS_GREATER_OR_EQUAL expr
            | expr INSTANCEOF expr'''
    p[0] = ast.BinaryOp(p[2].lower(), p[1], p[3], lineno=p.lineno(2))
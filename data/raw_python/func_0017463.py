def p_expression_binop(p):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression EQUAL expression
                  | expression CONCAT expression
                  | expression SPLIT expression'''
    v = p[2]
    if v == '+':
        p[0] = PlusOp(p[1], p[3])
    elif v == '-':
        p[0] = MinusOp(p[1], p[3])
    elif v == '*':
        p[0] = MultiplyOp(p[1], p[3])
    elif v == '/':
        p[0] = DivideOp(p[1], p[3])
    elif v == '=':
        p[0] = EqualOp(p[1], p[3])
    elif v == settings.concat_operator:
        p[0] = ConcatenationOp(p[1], p[3])
    elif v == settings.separator_operator:
        p[0] = SplittingOp(p[1], p[3])
    elif v == settings.field_operator:
        p[0] = Symbol(p[1], field=p[3])
def p_namespace_name(p):
    '''namespace_name : namespace_name NS_SEPARATOR STRING
                      | STRING'''
    if len(p) == 4:
        p[0] = p[1] + p[2] + p[3]
    else:
        p[0] = p[1]
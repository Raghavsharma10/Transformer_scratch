def p_class_name(p):
    '''class_name : namespace_name
                  | NS_SEPARATOR namespace_name
                  | NAMESPACE NS_SEPARATOR namespace_name'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 3:
        p[0] = p[1] + p[2]
    else:
        p[0] = p[1] + p[2] + p[3]
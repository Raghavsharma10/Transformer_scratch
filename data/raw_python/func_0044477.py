def p_operation_list(p):
    """
    operation_list : define_operation operation_list
                   | define_operation
    """
    if len(p) == 3:
        p[0] = p[1] + p[2]
    elif len(p) == 2:
        p[0] = p[1]
    else:
        raise RuntimeError("Invalid production rules 'p_operation_list'")
def p_definition_list(p):
    """
    definition_list : definition definition_list
                    | definition
    """
    if len(p) == 3:
        p[0] = p[1] + p[2]
    elif len(p) == 2:
        p[0] = p[1]
    else:
        raise RuntimeError("Invalid production rules 'p_action_list'")
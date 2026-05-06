def p_setting_list(p):
    """
    setting_list : setting setting_list
                 | setting
    """
    if len(p) == 3:
        p[0] = merge_map(p[1], p[2])
    elif len(p) == 2:
        p[0] = p[1]
    else:
        raise RuntimeError("Invalid production rules 'p_setting_list'")
def remove_getdisplay(field_name):
    '''
    for string 'get_FIELD_NAME_display' return 'FIELD_NAME'
    '''
    str_ini = 'get_'
    str_end = '_display'
    if str_ini == field_name[0:len(str_ini)] and str_end == field_name[(-1) * len(str_end):]:
        field_name = field_name[len(str_ini):(-1) * len(str_end)]
    return field_name
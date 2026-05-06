def format_enum(enum_def):
    '''Returns a cython enum from a :attr:`EnumSpec` instance.
    '''
    text = []
    text.append('cdef enum {}:'.format(enum_def.tp_name))
    for member in enum_def.values:
        if member.value:
            text.append('{}{} = {}'.format(tab, *member))
        else:
            text.append('{}{}'.format(tab, member.name))

    for name in enum_def.names:
        text.append('ctypedef {} {}'.format(enum_def.tp_name, name))

    return text
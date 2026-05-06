def format_struct(struct_def):
    '''Returns a cython struct from a :attr:`StructSpec` instance.
    '''
    text = []
    text.append('cdef struct {}:'.format(struct_def.tp_name))
    text.extend(
        ['{}{}'.format(tab, format_variable(var))
         for var in struct_def.members]
    )

    for name in struct_def.names:
        text.append('ctypedef {} {}'.format(struct_def.tp_name, name))

    return text
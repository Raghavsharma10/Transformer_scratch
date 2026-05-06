def corba_name_to_string(name):
    '''Convert a CORBA CosNaming.Name to a string.'''
    parts = []
    if type(name) is not list and type(name) is not tuple:
        raise NotCORBANameError(name)
    if len(name) == 0:
        raise NotCORBANameError(name)

    for nc in name:
        if not nc.kind:
            parts.append(nc.id)
        else:
            parts.append('{0}.{1}'.format(nc.id, nc.kind))
    return '/'.join(parts)
def thrift_print(structure, offset=0):
    """
    Handy recursive text ouput for thrift structures
    """
    if not is_thrift_item(structure):
        return str(structure)
    s = str(structure.__class__) + '\n'
    for key in dir(structure):
        if key.startswith('_') or key in ['thrift_spec', 'read', 'write',
                                          'default_spec', 'validate']:
            continue
        s = s + ' ' * offset + key + ': ' + thrift_print(getattr(structure, key)
                                                         , offset+2) + '\n'
    return s
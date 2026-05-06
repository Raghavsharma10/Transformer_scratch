def check_enum(enum, name=None, valid=None):
    """ Get lowercase string representation of enum.
    """
    name = name or 'enum'
    # Try to convert
    res = None
    if isinstance(enum, int):
        if hasattr(enum, 'name') and enum.name.startswith('GL_'):
            res = enum.name[3:].lower()
    elif isinstance(enum, string_types):
        res = enum.lower()
    # Check
    if res is None:
        raise ValueError('Could not determine string represenatation for'
                         'enum %r' % enum)
    elif valid and res not in valid:
        raise ValueError('Value of %s must be one of %r, not %r' % 
                         (name, valid, enum))
    return res
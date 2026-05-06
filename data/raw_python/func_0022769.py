def as_enum(enum):
    """ Turn a possibly string enum into an integer enum.
    """
    if isinstance(enum, string_types):
        try:
            enum = getattr(gl, 'GL_' + enum.upper())
        except AttributeError:
            try:
                enum = _internalformats['GL_' + enum.upper()]
            except KeyError:
                raise ValueError('Could not find int value for enum %r' % enum)
    return enum
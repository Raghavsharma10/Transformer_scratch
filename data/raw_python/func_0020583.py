def encode_args(args, extra=False):
    """
    Encode a list of arguments
    """
    if not args:
        return ''

    methodargs = ', '.join([encode(a) for a in args])
    if extra:
        methodargs += ', '

    return methodargs
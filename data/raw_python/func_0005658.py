def recode_unicode(s, encoding='utf-8'):
    """ Inputs are encoded to utf-8 and then decoded to the desired
        output encoding

        @encoding: the desired encoding

        -> #str with the desired @encoding
    """
    if isinstance(s, str):
        return s.encode().decode(encoding)
    return s
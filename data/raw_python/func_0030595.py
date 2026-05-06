def _to_ascii(s):
    """ Converts given string to ascii ignoring non ascii.
    Args:
        s (text or binary):

    Returns:
        str:
    """
    # TODO: Always use unicode within ambry.
    from six import text_type, binary_type
    if isinstance(s, text_type):
        ascii_ = s.encode('ascii', 'ignore')
    elif isinstance(s, binary_type):
        ascii_ = s.decode('utf-8').encode('ascii', 'ignore')
    else:
        raise Exception('Unknown text type - {}'.format(type(s)))
    return ascii_
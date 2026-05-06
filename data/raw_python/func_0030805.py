def _parse_binary(v, header_d):
    """ Parses binary string.

    Note:
        <str> for py2 and <binary> for py3.

    """

    # This is often a no-op, but it ocassionally converts numbers into strings

    v = nullify(v)

    if v is None:
        return None

    if six.PY2:
        try:
            return six.binary_type(v).strip()
        except UnicodeEncodeError:
            return six.text_type(v).strip()
    else:
        # py3
        try:
            return six.binary_type(v, 'utf-8').strip()
        except UnicodeEncodeError:
            return six.text_type(v).strip()
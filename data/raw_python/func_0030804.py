def _parse_text(v, header_d):
    """ Parses unicode.

    Note:
        unicode types for py2 and str types for py3.

    """

    v = nullify(v)

    if v is None:
        return None

    try:
        return six.text_type(v).strip()
    except Exception as e:
        raise CastingError(six.text_type, header_d, v, str(e))
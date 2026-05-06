def parse_obj(o):
    """
    Parses a given dictionary with the key being the OBD PID and the value its
    returned value by the OBD interface
    :param dict o:
    :return:
    """
    r = {}
    for k, v in o.items():
        if is_unable_to_connect(v):
            r[k] = None

        try:
            r[k] = parse_value(k, v)
        except (ObdPidParserUnknownError, AttributeError, TypeError):
            r[k] = None
    return r
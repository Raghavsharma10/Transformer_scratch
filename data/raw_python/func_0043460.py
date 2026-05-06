def parse_value(type: str, val: str):
    """
    Parses a given OBD value of a given type (PID)
    and returns the parsed value.
    If the PID is unknown / not implemented a PIDParserUnknownError
    will be raised including the type which was unknown
    :param type:
    :param val:
    :return:
    """
    if type.upper() in PARSER_MAP:
        #prep_val = prepare_value(val)
        out = PARSER_MAP[type](val)
        log.debug('For {} entered {}, got {} out'.format(type, val, out))
        return out
    else:
        raise ObdPidParserUnknownError(type, val)
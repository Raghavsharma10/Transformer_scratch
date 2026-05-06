def read_xml(filename):
    """
    Use et to read in a xml file, or string, into a Element object.

    :param filename: File to parse.
    :return: lxml._elementTree object or None
    """
    parser = et.XMLParser(remove_blank_text=True)
    isfile=False
    try:
        isfile = os.path.exists(filename)
    except ValueError as e:
        if 'path too long for Windows' in str(e):
            pass
        else:
            raise
    try:
        if isfile:
            return et.parse(filename, parser)
        else:
            r = et.fromstring(filename, parser)
            return r.getroottree()
    except IOError:
        log.exception('unable to open file [[}]'.format(filename))
    except et.XMLSyntaxError:
        log.exception('unable to parse XML [{}]'.format(filename))
        return None
    return None
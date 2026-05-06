def generous_parse_uri(uri):
    """Return a urlparse.ParseResult object with the results of parsing the
    given URI. This has the same properties as the result of parse_uri.

    When passed a relative path, it determines the absolute path, sets the
    scheme to file, the netloc to localhost and returns a parse of the result.
    """

    parse_result = urlparse(uri)

    if parse_result.scheme == '':
        abspath = os.path.abspath(parse_result.path)
        if IS_WINDOWS:
            abspath = windows_to_unix_path(abspath)
        fixed_uri = "file://{}".format(abspath)
        parse_result = urlparse(fixed_uri)

    return parse_result
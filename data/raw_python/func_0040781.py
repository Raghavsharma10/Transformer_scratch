def parseLsPermissionText(permission_text):
    """
    parse "ls -l" style permission text: e.g. -rw-r--r--
    """

    from six.moves import range

    match = re.search("[-drwx]+", permission_text)
    if match is None:
        raise ValueError(
            "invalid permission character: " + permission_text)

    if len(permission_text) != 10:
        raise ValueError(
            "invalid permission text length: " + permission_text)

    permission_text = permission_text[1:]

    return int(
        "0" + "".join([
            str(parsePermission3Char(permission_text[i:i + 3]))
            for i in range(0, 9, 3)
        ]),
        base=8)
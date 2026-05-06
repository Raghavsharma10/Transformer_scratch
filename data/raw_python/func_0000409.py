def escape_dn_chars(s):
    """
    Escape all DN special characters found in s
    with a back-slash (see RFC 4514, section 2.4)
    """
    if s:
        assert isinstance(s, six.string_types)
        s = s.replace('\\', '\\\\')
        s = s.replace(',', '\\,')
        s = s.replace('+', '\\+')
        s = s.replace('"', '\\"')
        s = s.replace('<', '\\<')
        s = s.replace('>', '\\>')
        s = s.replace(';', '\\;')
        s = s.replace('=', '\\=')
        s = s.replace('\000', '\\\000')
        if s[0] == '#' or s[0] == ' ':
            s = ''.join(('\\', s))
        if s[-1] == ' ':
            s = ''.join((s[:-1], '\\ '))
    return s
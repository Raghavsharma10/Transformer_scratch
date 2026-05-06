def remove_whitespace(s):
    """ Unsafely attempts to remove HTML whitespace. This is not an HTML parser
        which is why its considered 'unsafe', but it should work for most
        implementations. Just use on at your own risk.

        @s: #str

        -> HTML with whitespace removed, ignoring <pre>, script, textarea and code
            tags
    """
    ignores = {}
    for ignore in html_ignore_whitespace_re.finditer(s):
        name = "{}{}{}".format(r"{}", uuid.uuid4(), r"{}")
        ignores[name] = ignore.group()
        s = s.replace(ignore.group(), name)
    s = whitespace_re(r' ', s).strip()
    for name, val in ignores.items():
        s = s.replace(name, val)
    return s
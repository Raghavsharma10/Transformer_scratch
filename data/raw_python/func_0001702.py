def shell_escape(s):
    r"""Given bl"a, returns "bl\\"a".
    """
    if isinstance(s, PosixPath):
        s = unicode_(s)
    elif isinstance(s, bytes):
        s = s.decode('utf-8')
    if not s or any(c not in safe_shell_chars for c in s):
        return '"%s"' % (s.replace('\\', '\\\\')
                          .replace('"', '\\"')
                          .replace('`', '\\`')
                          .replace('$', '\\$'))
    else:
        return s
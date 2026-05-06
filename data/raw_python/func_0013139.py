def escape_windows_cmd_string(s):
    """Returns a string that is usable by the Windows cmd.exe.
    The escaping is based on details here and emperical testing:
    http://www.robvanderwoude.com/escapechars.php
    """
    for c in '()%!^<>&|"':
        s = s.replace(c, '^' + c)
    s = s.replace('/?', '/.')
    return s
def _guess_fmt_from_bytes(inp):
    """Try to guess format of given bytestring.

    Args:
        inp: byte string to guess format of
    Returns:
        guessed format
    """
    stripped = inp.strip()
    fmt = None
    ini_section_header_re = re.compile(b'^\[([\w-]+)\]')

    if len(stripped) == 0:
        # this can be anything, so choose yaml, for example
        fmt = 'yaml'
    else:
        if stripped.startswith(b'<'):
            fmt = 'xml'
        else:
            for l in stripped.splitlines():
                line = l.strip()
                # there are C-style comments in json5, but we don't auto-detect it,
                #  so it doesn't matter here
                if not line.startswith(b'#') and line:
                    break
            # json, ini or yaml => skip comments and then determine type
            if ini_section_header_re.match(line):
                fmt = 'ini'
            else:
                # we assume that yaml is superset of json
                # TODO: how do we figure out it's not yaml?
                fmt = 'yaml'

    return fmt
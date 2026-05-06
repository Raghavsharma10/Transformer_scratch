def parse_docstring(docstring):
    """
    Parse a PEP-257 docstring.

    SHORT -> blank line -> LONG

    """
    short_desc = long_desc = ''
    if docstring:
        docstring = trim(docstring.lstrip('\n'))
        lines = docstring.split('\n\n', 1)
        short_desc = lines[0].strip().replace('\n', ' ')

        if len(lines) > 1:
            long_desc = lines[1].strip()
    return short_desc, long_desc
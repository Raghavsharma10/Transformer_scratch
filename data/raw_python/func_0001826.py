def iexpand(string, keep_escapes=False):
    """Expand braces and return an iterator."""

    if isinstance(string, bytes):
        is_bytes = True
        string = string.decode('latin-1')

    else:
        is_bytes = False

    if is_bytes:
        return (entry.encode('latin-1') for entry in ExpandBrace(keep_escapes).expand(string))

    else:
        return (entry for entry in ExpandBrace(keep_escapes).expand(string))
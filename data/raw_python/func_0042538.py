def _scrub_key(key):
    """
    RETURN JUST THE :. CHARACTERS
    """
    if key == None:
        return None

    output = []
    for c in key:
        if c in [":", "."]:
            output.append(c)
    return "".join(output)
def toPairTreePath(name):
    """Cleans a string, and then splits it into a pairtree path."""
    sName = sanitizeString(name)
    chunks = []
    for x in range(0, len(sName)):
        if x % 2:
            continue
        if (len(sName) - 1) == x:
            chunk = sName[x]
        else:
            chunk = sName[x: x + 2]
        chunks.append(chunk)
    return os.sep.join(chunks) + os.sep
def deSanitizeString(name):
    """Reverses sanitization process.

    Reverses changes made to a string that has been sanitized for use
    as a pairtree identifier.
    """
    oldString = name
    # first pass
    replaceTable2 = [
        ("/", "="),
        (":", "+"),
        (".", ","),
    ]
    for r in replaceTable2:
        oldString = oldString.replace(r[1], r[0])
    # reverse ascii 0-32 stuff
    # must subtract number added at sanitization
    for x in range(0, 33):
        oldString = oldString.replace(
            hex(x + sanitizerNum).replace('0x', '^'), chr(x))
    # second pass
    replaceTable = [
        ('"', '^22'),
        ('<', '^3c'),
        ('?', '^3f'),
        ('*', '^2a'),
        ('=', '^3d'),
        ('+', '^2b'),
        ('>', '^3e'),
        ('|', '^7c'),
        (',', '^2c'),
        ('^', '^5e'),
    ]

    for r in replaceTable:
        oldString = oldString.replace(r[1], r[0])
    return oldString
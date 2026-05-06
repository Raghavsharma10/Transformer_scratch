def sanitizeString(name):
    """Cleans string in preparation for splitting for use as a pairtree
    identifier."""
    newString = name
    # string cleaning, pass 1
    replaceTable = [
        ('^', '^5e'),  # we need to do this one first
        ('"', '^22'),
        ('<', '^3c'),
        ('?', '^3f'),
        ('*', '^2a'),
        ('=', '^3d'),
        ('+', '^2b'),
        ('>', '^3e'),
        ('|', '^7c'),
        (',', '^2c'),
    ]

    #   "   hex 22           <   hex 3c           ?   hex 3f
    #   *   hex 2a           =   hex 3d           ^   hex 5e
    #   +   hex 2b           >   hex 3e           |   hex 7c
    #   ,   hex 2c

    for r in replaceTable:
        newString = newString.replace(r[0], r[1])
    # replace ascii 0-32
    for x in range(0, 33):
        # must add somewhat arbitrary num to avoid conflict at deSanitization
        # conflict example: is ^x1e supposed to be ^x1 (ascii 1) followed by
        # letter 'e' or really ^x1e (ascii 30)
        newString = newString.replace(
            chr(x), hex(x + sanitizerNum).replace('0x', '^'))

    replaceTable2 = [
        ("/", "="),
        (":", "+"),
        (".", ","),
    ]

    # / -> =
    # : -> +
    # . -> ,

    # string cleaning pass 2
    for r in replaceTable2:
        newString = newString.replace(r[0], r[1])
    return newString
def quote(text, ws=plain):
    """Quote special characters in shell command arguments.

    E.g ``--foo bar>=10.1`` becomes "--foo bar\>\=10\.1``.

    """
    return "".join(chr in ws and chr or '\\' + chr
                        for chr in text)
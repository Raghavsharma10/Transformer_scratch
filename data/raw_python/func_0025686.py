def Expand(instring, noerror=0):
    """
    Expand a string with embedded IRAF variables (IRAF virtual filename).

    Allows comma-separated lists.  Also uses os.path.expanduser to replace '~'
    symbols.

    Set the noerror flag to silently replace undefined variables with just the
    variable name or null (so Expand('abc$def') = 'abcdef' and
    Expand('(abc)def') = 'def').  This is the IRAF behavior, though it is
    confusing and hides errors.
    """

    # call _expand1 for each entry in comma-separated list
    wordlist = instring.split(",")
    outlist = []
    for word in wordlist:
        outlist.append(os.path.expanduser(_expand1(word, noerror=noerror)))
    return ",".join(outlist)
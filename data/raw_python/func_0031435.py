def _makeComplementTable(complementData):
    """
    Make a sequence complement table.

    @param complementData: A C{dict} whose keys and values are strings of
        length one. A key, value pair indicates a substitution that should
        be performed during complementation.
    @return: A 256 character string that can be used as a translation table
        by the C{translate} method of a Python string.
    """
    table = list(range(256))
    for _from, to in complementData.items():
        table[ord(_from[0].lower())] = ord(to[0].lower())
        table[ord(_from[0].upper())] = ord(to[0].upper())
    return ''.join(map(chr, table))
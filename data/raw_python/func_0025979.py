def csvSplit(line, delim=',', allowEol=True):
    """ Take a string as input (e.g. a line in a csv text file), and break
    it into tokens separated by commas while ignoring commas embedded inside
    quoted sections.  This is exactly what the 'csv' module is meant for, so
    we *should* be using it, save that it has two bugs (described next) which
    limit our use of it.  When these bugs are fixed, this function should be
    forsaken in favor of direct use of the csv module (or similar).

    The basic use case is to split a function signature string, so for:
        afunc(arg1='str1', arg2='str, with, embedded, commas', arg3=7)
    we want a 3 element sequence:
        ["arg1='str1'", "arg2='str, with, embedded, commas'", "arg3=7"]

    but:
    >>> import csv
    >>> y = "arg1='str1', arg2='str, with, embedded, commas', arg3=7"
    >>> rdr = csv.reader( (y,), dialect='excel', quotechar="'", skipinitialspace=True)
    >>> l = rdr.next(); print(len(l), str(l))  # doctest: +SKIP
    6 ["arg1='str1'", "arg2='str", 'with', 'embedded', "commas'", "arg3=7"]

    which we can see is not correct - we wanted 3 tokens.  This occurs in
    Python 2.5.2 and 2.6.  It seems to be due to the text at the start of each
    token ("arg1=") i.e. because the quote isn't for the whole token.  If we
    were to remove the names of the args and the equal signs, it works:

    >>> x = "'str1', 'str, with, embedded, commas', 7"
    >>> rdr = csv.reader( (x,), dialect='excel', quotechar="'", skipinitialspace=True)
    >>> l = rdr.next(); print(len(l), str(l))  # doctest: +SKIP
    3 ['str1', 'str, with, embedded, commas', '7']

    But even this usage is delicate - when we turn off skipinitialspace, it
    fails:

    >>> x = "'str1', 'str, with, embedded, commas', 7"
    >>> rdr = csv.reader( (x,), dialect='excel', quotechar="'")
    >>> l = rdr.next(); print(len(l), str(l))  # doctest: +SKIP
    6 ['str1', " 'str", ' with', ' embedded', " commas'", ' 7']

    So, for now, we'll roll our own.
    """
    # Algorithm:  read chars left to right, go from delimiter to delimiter,
    # but as soon as a single/double/triple quote is hit, scan forward
    # (ignoring all else) until its matching end-quote is found.
    # For now, we will not specially handle escaped quotes.
    tokens = []
    ldl = len(delim)
    keepOnRollin = line is not None and len(line) > 0
    while keepOnRollin:
        tok = _getCharsUntil(line, delim, True, allowEol=allowEol)
        # len of token should always be > 0 because it includes end delimiter
        # except on last token
        if len(tok) > 0:
            # append it, but without the delimiter
            if tok[-ldl:] == delim:
                tokens.append(tok[:-ldl])
            else:
                tokens.append(tok) # tok goes to EOL - has no delimiter
                keepOnRollin = False
            line = line[len(tok):]
        else:
            # This is the case of the empty end token
            tokens.append('')
            keepOnRollin = False
    return tokens
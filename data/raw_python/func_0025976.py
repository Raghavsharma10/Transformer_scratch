def printColsAuto(in_strings, term_width=80, min_pad=1):
    """ Print a list of strings centered in columns.  Determine the number
    of columns and lines on the fly.  Return the result, ready to print.
    in_strings is a list/tuple/iterable of strings
    min_pad is number of spaces to appear on each side of a single string (so
            you will see twice this many spaces between 2 strings)
    """
    # sanity check
    assert in_strings and len(in_strings)>0, 'Unexpected: '+repr(in_strings)

    # get max width in input
    maxWidth = len(max(in_strings, key=len)) + (2*min_pad) # width with pad
    numCols = term_width//maxWidth # integer div
    # set numCols so we take advantage of the whole line width
    numCols = min(numCols, len(in_strings))

    # easy case - single column or too big
    if numCols < 2:
        # one or some items are too big but print one item per line anyway
        lines = [x.center(term_width) for x in in_strings]
        return '\n'.join(lines)

    # normal case - 2 or more columns
    colWidth = term_width//numCols # integer div
    # colWidth is guaranteed to be larger than all items in input
    retval = ''
    for i in range(len(in_strings)):
        retval+=in_strings[i].center(colWidth)
        if (i+1)%numCols == 0:
            retval += '\n'
    return retval.rstrip()
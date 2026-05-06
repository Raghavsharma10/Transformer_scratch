def parseRangeString(s, convertToZeroBased=False):
    """
    Parse a range string of the form 1-5,12,100-200.

    @param s: A C{str} specifiying a set of numbers, given in the form of
        comma separated numeric ranges or individual indices.
    @param convertToZeroBased: If C{True} all indices will have one
        subtracted from them.
    @return: A C{set} of all C{int}s in the specified set.
    """

    result = set()
    for _range in s.split(','):
        match = _rangeRegex.match(_range)
        if match:
            start, end = match.groups()
            start = int(start)
            if end is None:
                end = start
            else:
                end = int(end)
            if start > end:
                start, end = end, start
            if convertToZeroBased:
                result.update(range(start - 1, end))
            else:
                result.update(range(start, end + 1))
        else:
            raise ValueError(
                'Illegal range %r. Ranges must single numbers or '
                'number-number.' % _range)

    return result
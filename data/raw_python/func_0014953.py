def romanNumeral(n):
    """
    >>> romanNumeral(13)
    'XIII'
    >>> romanNumeral(2944)
    'MMCMXLIV'
    """
    if 0 > n > 4000: raise ValueError('``n`` must lie between 1 and 3999: %d' % n)
    roman   = 'I IV  V  IX  X  XL   L  XC    C   CD    D   CM     M'.split()
    arabic  = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    res = []
    while n>0:
        pos = bisect.bisect_right(arabic, n)-1
        fit = n//arabic[pos]
        res.append(roman[pos]*fit); n -= fit * arabic[pos]
    return "".join(res)
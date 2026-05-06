def chr22XY(c):
    """force to name from 1..22, 23, 24, X, Y, M 
    to in chr1..chr22, chrX, chrY, chrM
    str or ints accepted

    >>> chr22XY('1')
    'chr1'
    >>> chr22XY(1)
    'chr1'
    >>> chr22XY('chr1')
    'chr1'
    >>> chr22XY(23)
    'chrX'
    >>> chr22XY(24)
    'chrY'
    >>> chr22XY("X")
    'chrX'
    >>> chr22XY("23")
    'chrX'
    >>> chr22XY("M")
    'chrM'

    """
    c = str(c)
    if c[0:3] == 'chr':
        c = c[3:]
    if c == '23':
        c = 'X'
    if c == '24':
        c = 'Y'
    return 'chr' + c
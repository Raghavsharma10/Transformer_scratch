def fbpe_key(code):
    """
    input:
        'S0102-67202009000300001'
    output:
        'S0102-6720(09)000300001'
    """

    begin = code[0:10]
    year = code[12:14]
    end = code[14:]

    return '%s(%s)%s' % (begin, year, end)
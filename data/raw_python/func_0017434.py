def polygen(*coefficients):
    '''Polynomial generating function'''
    if not coefficients:
        return lambda i: 0
    else:
        c0 = coefficients[0]
        coefficients = coefficients[1:]

        def _(i):
            v = c0
            for c in coefficients:
                v += c*i
                i *= i
            return v

        return _
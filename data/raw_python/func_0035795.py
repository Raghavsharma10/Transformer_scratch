def parse_ion_string(istr, analyses, twin=None):
    """
    Recursive string parser that handles "ion" strings.
    """

    if istr.strip() == '':
        return Trace()

    # remove (unnessary?) pluses from the front
    # TODO: plus should be abs?
    istr = istr.lstrip('+')

    # invert it if preceded by a minus sign
    if istr[0] == '-':
        return -parse_ion_string(istr[1:], analyses, twin)

    # this is a function or paranthesized expression
    if is_parans_exp(istr):
        if ')' not in istr:
            # unbalanced parantheses
            pass
        fxn = istr.split('(')[0]
        args = istr[istr.find('(') + 1:istr.find(')')].split(',')
        if fxn == '':
            # strip out the parantheses and continue
            istr = args[0]
        else:
            ts = parse_ion_string(args[0], analyses, twin)
            # FIXME
            return ts
            # return fxn_resolver(ts, fxn, *args[1:])

    # all the complicated math is gone, so simple lookup
    if set(istr).intersection(set('+-/*()')) == set():
        if istr in SHORTCUTS:
            # allow some shortcuts to pull out common ions
            return parse_ion_string(SHORTCUTS[istr], analyses, twin)
        elif istr[0] == '!' and all(i in '0123456789.' for i in istr[1:]):
            # TODO: should this handle negative numbers?
            return float(istr[1:])
        elif istr == '!pi':
            return np.pi
        elif istr == '!e':
            return np.e
        else:
            return trace_resolver(istr, analyses, twin)

    # go through and handle operators
    for token in '/*+-^':
        if len(tokenize(istr, token)) != 1:
            ts = tokenize(istr, token)
            s = parse_ion_string(ts[0], analyses, twin)
            for t in ts[1:]:
                if token == '/':
                    s /= parse_ion_string(t, analyses, twin)
                elif token == '*':
                    s *= parse_ion_string(t, analyses, twin)
                elif token == '+':
                    s += parse_ion_string(t, analyses, twin)
                elif token == '-':
                    s -= parse_ion_string(t, analyses, twin)
                elif token == '^':
                    s **= parse_ion_string(t, analyses, twin)
            return s
    raise Exception('Parser hit a point it shouldn\'t have!')
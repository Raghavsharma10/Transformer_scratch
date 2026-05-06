def deflood(s, n=3):
    """ Returns the string with no more than n repeated characters, e.g.,
        deflood("NIIIICE!!", n=1) => "Nice!"
        deflood("nice.....", n=3) => "nice..."
    """
    if n == 0:
        return s[0:0]
    return re.sub(r"((.)\2{%s,})" % (n-1), lambda m: m.group(1)[0] * n, s)
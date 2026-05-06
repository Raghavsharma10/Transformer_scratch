def random_id(length=16, charset=alphanum_chars, first_charset=alpha_chars, sep='', group=0):
    """Creates a random id with the given length and charset.
    ## Parameters
    * length          the number of characters in the id
    * charset         what character set to use (a list of characters)
    * first_charset   what character set for the first character
    * sep=''          what character to insert between groups
    * group=0         how long the groups are (default 0 means no groups)
    """
    t = []

    first_chars = list(set(charset).intersection(first_charset))
    if len(first_chars) == 0:
        first_chars = charset

    t.append(first_chars[random.randrange(len(first_chars))])

    for i in range(len(t), length):
        if (group > 0) and (i % group == 0) and (i < length):
            t.append(sep)
        t.append(charset[random.randrange(len(charset))])

    return ''.join(t)
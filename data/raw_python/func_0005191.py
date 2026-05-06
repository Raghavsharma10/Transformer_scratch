def is_username(string, minlen=1, maxlen=15):
    """ Determines whether the @string pattern is username-like
        @string: #str being tested
        @minlen: minimum required username length
        @maxlen: maximum username length

        -> #bool
    """
    if string:
        string = string.strip()
        return username_re.match(string) and (minlen <= len(string) <= maxlen)
    return False
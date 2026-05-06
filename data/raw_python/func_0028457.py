def multireplace(string,  # type: unicode
                 patterns,  # type: str_or_str_iterable
                 substitutions,  # type: str_istr_icallable
                 maxreplace=0,  # type: int
                 flags=0  # type: unicode
                 ):  # type: (...) -> bool
    """ Like unicode.replace() but accept several substitutions and regexes

        Args:
            string: the string to split on.
            patterns: a string, or an iterable of strings to be replaced.
            substitutions: a string or an iterable of string to use as a
                           replacement. You can pass either one string, or
                           an iterable containing the same number of
                           sustitutions that you passed as patterns. You can
                           also pass a callable instead of a string. It
                           should expact a match object as a parameter.
            maxreplace: the max number of replacement to make. 0 is no limit,
                        which is the default.
            flags: flags you wish to pass if you use regexes. You should
                   pass them as a string containing a combination of:

                    - 'm' for re.MULTILINE
                    - 'x' for re.VERBOSE
                    - 'v' for re.VERBOSE
                    - 's' for re.DOTALL
                    - '.' for re.DOTALL
                    - 'd' for re.DEBUG
                    - 'i' for re.IGNORECASE
                    - 'u' for re.UNICODE
                    - 'l' for re.LOCALE

        Returns:
            The string with replaced bits.

        Raises:
            ValueError: if you pass the wrong number of substitution.

        Example:

            >>> print(multireplace(u'a,b;c/d', (u',', u';', u'/'), u','))
            a,b,c,d
            >>> print(multireplace(u'a1b33c-d', u'\d+', u','))
            a,b,c-d
            >>> print(multireplace(u'a-1,b-3,3c-d', u',|-', u'', maxreplace=3))
            a1b3,3c-d
            >>> def upper(match):
            ...     return match.group().upper()
            ...
            >>> print(multireplace(u'a-1,b-3,3c-d', u'[ab]', upper))
            A-1,B-3,3c-d
    """

    # we can pass either a string or an iterable of strings
    patterns = ensure_tuple(patterns)
    substitutions = ensure_tuple(substitutions)

    # you can either have:
    # - many patterns, one substitution
    # - many patterns, exactly as many substitutions
    # anything else is an error
    num_of_subs = len(substitutions)
    num_of_patterns = len(patterns)

    if num_of_subs == 1 and num_of_patterns > 0:
        substitutions *= num_of_patterns
    elif len(patterns) != num_of_subs:
            raise ValueError("You must have exactly one substitution "
                             "for each pattern or only one substitution")

    flags = parse_re_flags(flags)

    # no limit for replacing, use a simple code
    if not maxreplace:
        for pattern, sub in zip(patterns, substitutions):
            string, count = re.subn(pattern, sub, string, flags=flags)
        return string

    # ensure we respect the max number of replace accross substitutions
    for pattern, sub in zip(patterns, substitutions):
        string, count = re.subn(pattern, sub, string,
                                count=maxreplace, flags=flags)
        maxreplace -= count
        if maxreplace == 0:
            break

    return string
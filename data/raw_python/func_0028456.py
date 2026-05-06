def multisplit(string,  # type: unicode
               *separators,  # type: unicode
               **kwargs  # type: Union[unicode, C[..., I[unicode]]]
               ):  # type: (...) -> I
    """ Like unicode.split, but accept several separators and regexes

        Args:
            string: the string to split.
            separators: strings you can split on. Each string can be a
                        regex.
            maxsplit: max number of time you wish to split. default is 0,
                      which means no limit.
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
            cast: what to cast the result to

        Returns:
            An iterable of substrings.

        Raises:
            ValueError: if you pass a flag without separators.
            TypeError: if you pass something else than unicode strings.

        Example:

            >>> for word in multisplit(u'fat     black cat, big'): print(word)
            fat
            black
            cat,
            big
            >>> string = u'a,b;c/d=a,b;c/d'
            >>> chunks = multisplit(string, u',', u';', u'[/=]', maxsplit=4)
            >>> for chunk in chunks: print(chunk)
            a
            b
            c
            d
            a,b;c/d

    """

    cast = kwargs.pop('cast', list)
    flags = parse_re_flags(kwargs.get('flags', 0))
    # 0 means "no limit" for re.split
    maxsplit = require_positive_number(kwargs.get('maxsplit', 0),
                                       'maxsplit')

    # no separator means we use the default unicode.split behavior
    if not separators:
        if flags:
            raise ValueError(ww.s >> """
                             You can't pass flags without passing
                             a separator. Flags only have sense if
                             you split using a regex.
                            """)

        maxsplit = maxsplit or -1  # -1 means "no limit" for unicode.split
        return unicode.split(string, None, maxsplit)

    # Check that all separators are strings
    for i, sep in enumerate(separators):
        if not isinstance(sep, unicode):
            raise TypeError(ww.s >> """
                '{!r}', the separator at index '{}', is of type '{}'.
                multisplit() only accepts unicode strings.
            """.format(sep, i, type(sep)))

    # TODO: split let many empty strings in the result. Fix it.

    seps = list(separators)  # cast to list so we can slice it

    # simple code for when you need to split the whole string
    if maxsplit == 0:
        return cast(_split(string, seps, flags))

    # slow implementation with checks for recursive maxsplit
    return cast(_split_with_max(string, seps, maxsplit, flags))
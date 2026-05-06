def tr(string1, string2, source, option=''):
    """Replace or remove specific characters.

    If not given option, then replace all characters in string1 with
    the character in the same position in string2.

    Following options are available:
        c   Replace all complemented characters in string1 with
            the character in the same position in string2.
        d   Delete all characters in string1.
        s   Squeeze all characters in string1.
        cs  Squeeze all the characters in string2 besides "c" replacement.
        ds  Delete all characters in string1. Squeeze all characters
            in string2.
        cd  Delete all complemented characters in string1.

    Params:
        <unicode> string1
        <unicode> string2
        <unicode> source
        <basestring> option
    Return:
        <unicode> translated_source
    """
    if not is_valid_type(source):
        raise TypeError('source must be unicode')

    from_list = make_char_list(string1)
    if option == 's':
        from_list = to_unichr(from_list)
        return squeeze(from_list, source)
    elif 'c' in option:
        from_list = to_unichr(from_list)
        from_list = [ord(c) for c in set(source) - set(from_list)]
        if 'd' in option:
            to_list = [None for i in from_list]
        else:
            to_list = [string2[-1] for i in from_list]
        source = translate(from_list, to_list, source)
        if 's' in option:
            source = squeeze(to_list, source)
        return source
    elif 'd' in option:
        to_list = [None for i in from_list]
        source = translate(from_list, to_list, source)
        if 's' in option:
            to_list = make_char_list(string2)
            to_list = to_unichr(to_list)
            source = squeeze(to_list, source)
        return source
    else:
        to_list = make_char_list(string2)
        length_diff = (len(from_list) - len(to_list))
        if length_diff:
            to_list += [to_list[-1]] * length_diff
        to_list = to_unichr(to_list)
        return translate(from_list, to_list, source)
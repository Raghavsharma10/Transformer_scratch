def append_comps(comps, char):
    """
    Append a character to `comps` following this rule: a vowel is added to the
    vowel part if there is no last consonant, else to the last consonant part;
    a consonant is added to the first consonant part if there is no vowel, and
    to the last consonant part if the vowel part is not empty.

    >>> transform(['', '', ''])
    ['c', '', '']
    >>> transform(['c', '', ''], '+o')
    ['c', 'o', '']
    >>> transform(['c', 'o', ''], '+n')
    ['c', 'o', 'n']
    >>> transform(['c', 'o', 'n'], '+o')
    ['c', 'o', 'no']
    """
    c = list(comps)
    if is_vowel(char):
        if not c[2]: pos = 1
        else: pos = 2
    else:
        if not c[2] and not c[1]: pos = 0
        else: pos = 2
    c[pos] += char
    return c
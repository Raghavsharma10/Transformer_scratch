def separate(string):
    """
    Separate a string into smaller parts: first consonant (or head), vowel,
    last consonant (if any).

    >>> separate('tuong')
    ['t','uo','ng']
    >>> separate('ohmyfkinggod')
    ['ohmyfkingg','o','d']
    """
    def atomic_separate(string, last_chars, last_is_vowel):
        if string == "" or (last_is_vowel != is_vowel(string[-1])):
            return (string, last_chars)
        else:
            return atomic_separate(string[:-1],
                                   string[-1] + last_chars, last_is_vowel)

    head, last_consonant = atomic_separate(string, "", False)
    first_consonant, vowel = atomic_separate(head, "", True)

    if last_consonant and not (vowel + first_consonant):
        comps = [last_consonant, '', '']  # ['', '', b] -> ['b', '', '']
    else:
        comps = [first_consonant, vowel, last_consonant]

    # 'gi' and 'qu' are considered qualified consonants.
    # We want something like this:
    #     ['g', 'ia', ''] -> ['gi', 'a', '']
    #     ['q', 'ua', ''] -> ['qu', 'a', '']
    if (comps[0] != '' and comps[1] != '') and \
        ((comps[0] in 'gG' and comps[1][0] in 'iI' and len(comps[1]) > 1) or
         (comps[0] in 'qQ' and comps[1][0] in 'uU')):
        comps[0] += comps[1][:1]
        comps[1] = comps[1][1:]

    return comps
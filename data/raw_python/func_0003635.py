def _tokenize_by_character_class(s):
    """
    Return a list of strings by splitting s (tokenizing) by character class.

    For example:
    _tokenize_by_character_class('Sat Jan 11 19:54:52 MST 2014') => ['Sat', ' ', 'Jan', ' ', '11', ' ', '19', ':',
        '54', ':', '52', ' ', 'MST', ' ', '2014']
    _tokenize_by_character_class('2013-08-14') => ['2013', '-', '08', '-', '14']
    """
    character_classes = [string.digits, string.ascii_letters, string.punctuation, string.whitespace]

    result = []
    rest = list(s)
    while rest:
        progress = False
        for character_class in character_classes:
            if rest[0] in character_class:
                progress = True
                token = ''
                for take_away in itertools.takewhile(lambda c: c in character_class, rest[:]):
                    token += take_away
                    rest.pop(0)
                result.append(token)
                break
        if not progress:  # none of the character classes matched; unprintable character?
            result.append(rest[0])
            rest = rest[1:]

    return result
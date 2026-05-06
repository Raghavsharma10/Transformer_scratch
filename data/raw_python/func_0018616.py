def multi_split(text, regexes):
    """
    Split the text by the given regexes, in priority order.

    Make sure that the regex is parenthesized so that matches are returned in
    re.split().

    Splitting on a single regex works like normal split.
    >>> '|'.join(multi_split('one two three', [r'\w+']))
    'one| |two| |three'

    Splitting on digits first separates the digits from their word
    >>> '|'.join(multi_split('one234five 678', [r'\d+', r'\w+']))
    'one|234|five| |678'

    Splitting on words first keeps the word with digits intact.
    >>> '|'.join(multi_split('one234five 678', [r'\w+', r'\d+']))
    'one234five| |678'
    """
    def make_regex(s):
        return re.compile(s) if isinstance(s, basestring) else s
    regexes = [make_regex(r) for r in regexes]

    # Run the list of pieces through the regex split, splitting it into more
    # pieces. Once a piece has been matched, add it to finished_pieces and
    # don't split it again. The pieces should always join back together to form
    # the original text.
    piece_list = [text]
    finished_pieces = set()
    def apply_re(regex, piece_list):
        for piece in piece_list:
            if piece in finished_pieces:
                yield piece
                continue
            for s in full_split(piece, regex):
                if regex.match(s):
                    finished_pieces.add(s)
                if s:
                    yield s

    for regex in regexes:
        piece_list = list(apply_re(regex, piece_list))
        assert ''.join(piece_list) == text
    return piece_list
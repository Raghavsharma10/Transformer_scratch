def number_aware_alphabetical_cmp(str1, str2):
    """ cmp function for sorting a list of strings by alphabetical order, but with
        numbers sorted numerically.

        i.e., foo1, foo2, foo10, foo11
        instead of foo1, foo10
    """

    def flatten_tokens(tokens):
        l = []
        for token in tokens:
            if isinstance(token, str):
                for char in token:
                    l.append(char)
            else:
                assert isinstance(token, float)
                l.append(token)
        return l

    seq1 = flatten_tokens(tokenize_by_number(str1))
    seq2 = flatten_tokens(tokenize_by_number(str2))

    l = min(len(seq1),len(seq2))

    i = 0

    while i < l:
        if seq1[i] < seq2[i]:
            return -1
        elif seq1[i] > seq2[i]:
            return 1
        i += 1

    if len(seq1) < len(seq2):
        return -1
    elif len(seq1) > len(seq2):
        return 1

    return 0
def split_using_dictionary(string, dictionary, max_key_length, single_char_parsing=False):
    """
    Return a list of (non-empty) substrings of the given string,
    where each substring is either:
    
    1. the longest string starting at the current index
       that is a key in the dictionary, or
    2. a single character that is not a key in the dictionary.

    If ``single_char_parsing`` is ``False``,
    parse the string one Unicode character at a time,
    that is, do not perform the greedy parsing.

    :param iterable string: the iterable object ("string") to split into atoms
    :param dict dictionary: the dictionary mapping atoms ("characters") to something else
    :param int max_key_length: the length of a longest key, in number of characters
    :param bool single_char_parsing: if ``True``, parse one Unicode character at a time
    """
    def substring(string, i, j):
        if isinstance(string[i], tuple):
            # transform list of tuples with one element in a tuple with all elements
            return tuple([string[k][0] for k in range(i, j)])
        # just return substring
        return string[i:j]
    
    if string is None:
        return None
    if (single_char_parsing) or (max_key_length < 2):
        return [c for c in string]
    acc = []
    l = len(string)
    i = 0
    while i < l:
        found = False
        for j in range(min(i + max_key_length, l), i, -1):
            sub = substring(string, i, j)
            if sub in dictionary:
                found = True
                acc.append(sub)
                i = j
                break
        if not found:
            acc.append(string[i])
            i += 1
    return acc
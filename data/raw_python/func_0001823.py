def get_matching_text_in_strs(a, b, match_min_size=30, ignore='', end_characters=''):
    # type: (str, str, int, str, str) -> List[str]
    """Returns a list of matching blocks of text in a and b

    Args:
        a (str): First string to match
        b (str): Second string to match
        match_min_size (int): Minimum block size to match on. Defaults to 30.
        ignore (str): Any characters to ignore in matching. Defaults to ''.
        end_characters (str): End characters to look for. Defaults to ''.

    Returns:
        List[str]: List of matching blocks of text

    """
    compare = difflib.SequenceMatcher(lambda x: x in ignore)
    compare.set_seqs(a=a, b=b)
    matching_text = list()

    for match in compare.get_matching_blocks():
        start = match.a
        text = a[start: start+match.size]
        if end_characters:
            prev_text = text
            while len(text) != 0 and text[0] in end_characters:
                text = text[1:]
            while len(text) != 0 and text[-1] not in end_characters:
                text = text[:-1]
            if len(text) == 0:
                text = prev_text
        if len(text) >= match_min_size:
            matching_text.append(text)
    return matching_text
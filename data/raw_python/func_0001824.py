def get_matching_text(string_list, match_min_size=30, ignore='', end_characters='.!\r\n'):
    # type: (List[str], int, str, str) -> str
    """Returns a string containing matching blocks of text in a list of strings followed by non-matching.

    Args:
        string_list (List[str]): List of strings to match
        match_min_size (int): Minimum block size to match on. Defaults to 30.
        ignore (str): Any characters to ignore in matching. Defaults to ''.
        end_characters (str): End characters to look for. Defaults to '.\r\n'.

    Returns:
        str: String containing matching blocks of text followed by non-matching

    """
    a = string_list[0]
    for i in range(1, len(string_list)):
        b = string_list[i]
        result = get_matching_text_in_strs(a, b, match_min_size=match_min_size, ignore=ignore,
                                           end_characters=end_characters)
        a = ''.join(result)
    return a
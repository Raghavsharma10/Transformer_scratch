def is_pangram(string):
    """
    Checks if the string is a pangram (https://en.wikipedia.org/wiki/Pangram).

    :param string: String to check.
    :type string: str
    :return: True if the string is a pangram, False otherwise.
    """
    return is_full_string(string) and set(SPACES_RE.sub('', string)).issuperset(letters_set)
def is_palindrome(string, strict=True):
    """
    Checks if the string is a palindrome (https://en.wikipedia.org/wiki/Palindrome).

    :param string: String to check.
    :type string: str
    :param strict: True if white spaces matter (default), false otherwise.
    :type strict: bool
    :return: True if the string is a palindrome (like "otto", or "i topi non avevano nipoti" if strict=False),
    False otherwise
    """
    if is_full_string(string):
        if strict:
            return reverse(string) == string
        return is_palindrome(SPACES_RE.sub('', string))
    return False
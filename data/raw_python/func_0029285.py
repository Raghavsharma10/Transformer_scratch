def shuffle(string):
    """
    Return a new string containing shuffled items.

    :param string: String to shuffle
    :type string: str
    :return: Shuffled string
    :rtype: str
    """
    s = sorted(string)  # turn the string into a list of chars
    random.shuffle(s)  # shuffle the list
    return ''.join(s)
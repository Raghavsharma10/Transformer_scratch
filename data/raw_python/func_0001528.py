def tokenize_words(string):
    """
    Tokenize input text to words.

    :param string: Text to tokenize
    :type string: str or unicode
    :return: words
    :rtype: list of strings
    """
    string = six.text_type(string)
    return re.findall(WORD_TOKENIZATION_RULES, string)
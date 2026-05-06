def tokenize_text(string):
    """
    Tokenize input text to paragraphs, sentences and words.

    Tokenization to paragraphs is done using simple Newline algorithm
    For sentences and words tokenizers above are used

    :param string: Text to tokenize
    :type string: str or unicode
    :return: text, tokenized into paragraphs, sentences and words
    :rtype: list of list of list of words
    """
    string = six.text_type(string)
    rez = []
    for part in string.split('\n'):
        par = []
        for sent in tokenize_sents(part):
            par.append(tokenize_words(sent))
        if par:
            rez.append(par)
    return rez
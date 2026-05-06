def tokenize_sents(string):
    """
    Tokenize input text to sentences.

    :param string: Text to tokenize
    :type string: str or unicode
    :return: sentences
    :rtype: list of strings
    """
    string = six.text_type(string)

    spans = []
    for match in re.finditer('[^\s]+', string):
        spans.append(match)
    spans_count = len(spans)

    rez = []
    off = 0

    for i in range(spans_count):
        tok = string[spans[i].start():spans[i].end()]
        if i == spans_count - 1:
            rez.append(string[off:spans[i].end()])
        elif tok[-1] in ['.', '!', '?', '…', '»']:
            tok1 = tok[re.search('[.!?…»]', tok).start()-1]
            next_tok = string[spans[i + 1].start():spans[i + 1].end()]
            if (next_tok[0].isupper()
                and not tok1.isupper()
                and not (tok[-1] != '.'
                         or tok1[0] == '('
                         or tok in ABBRS)):
                rez.append(string[off:spans[i].end()])
                off = spans[i + 1].start()

    return rez
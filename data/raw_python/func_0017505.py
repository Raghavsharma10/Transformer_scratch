def tokenize_init(spec):
    """Initialize a tokenizer. Should only be called by the
    :func:`~textparser.Parser.tokenize` method in the parser.

    """

    tokens = [Token('__SOF__', '__SOF__', 0)]
    re_token = '|'.join([
        '(?P<{}>{})'.format(name, regex) for name, regex in spec
    ])

    return tokens, re_token
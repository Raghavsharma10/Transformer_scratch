def txt2mecab(text, **kwargs):
    ''' Use mecab to parse one sentence '''
    mecab_out = _internal_mecab_parse(text, **kwargs).splitlines()
    tokens = [MeCabToken.parse(x) for x in mecab_out]
    return MeCabSent(text, tokens)
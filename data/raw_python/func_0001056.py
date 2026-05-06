def parse(content, *args, **kwargs):
    ''' Use mecab-python3 by default to parse JP text. Fall back to mecab binary app if needed '''
    global MECAB_PYTHON3
    if 'mecab_loc' not in kwargs and MECAB_PYTHON3 and 'MeCab' in globals():
        return MeCab.Tagger(*args).parse(content)
    else:
        return run_mecab_process(content, *args, **kwargs)
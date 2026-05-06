def lines2mecab(lines, **kwargs):
    ''' Use mecab to parse many lines '''
    sents = []
    for line in lines:
        sent = txt2mecab(line, **kwargs)
        sents.append(sent)
    return sents
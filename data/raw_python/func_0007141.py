def replace_umlauts(word, put_back=False):  # use translate()
    '''If put_back is True, put in umlauts; else, take them out!'''
    if put_back:
        word = word.replace('A', 'ä')
        word = word.replace('O', 'ö')

    else:
        word = word.replace('ä', 'A').replace('\xc3\xa4', 'A')
        word = word.replace('ö', 'O').replace('\xc3\xb6', 'O')

    return word
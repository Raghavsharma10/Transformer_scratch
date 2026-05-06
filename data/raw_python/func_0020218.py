def stemming_processor(words):
    '''Porter Stemmer word processor'''
    stem = PorterStemmer().stem
    for word in words:
        word = stem(word, 0, len(word)-1)
        yield word
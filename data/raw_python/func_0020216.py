def metaphone_processor(words):
    '''Double metaphone word processor.'''
    for word in words:
        for w in double_metaphone(word):
            if w:
                w = w.strip()
                if w:
                    yield w
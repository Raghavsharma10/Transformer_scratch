def translate_word(word, dictionary=['simplified']):
    '''
    Return the set of translations for a single character or word, if
    available.
    '''
    if not dictionaries:
        init()
    for d in dictionary:
        if word in dictionaries[d]:
            return dictionaries[d][word]
    return None
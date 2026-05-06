def all_phrase_translations(phrase):
    '''
    Return the set of translations for all possible words in a full
    phrase. Chinese is sometimes ambiguous. We do not attempt to
    disambiguate, or handle unknown letters especially well. Full
    parsing is left to upstream logic.
    '''
    if not trees:
        init()
    phrase = phrase.split(string.whitespace)
    for word in phrase:
        for x in range(len(word)):
            for translation in _words_at_the_beginning(
                    word[x+1:],
                    trees['simplified'][word[x]],
                    prefix=word[x]):
                yield translation
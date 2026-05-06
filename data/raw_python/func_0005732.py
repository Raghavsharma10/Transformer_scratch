def __replace_nouns(sentence, counts):
    """Lets find and replace all instances of #NOUN
    :param _sentence:
    :param counts:
    """

    if sentence is not None:
        while sentence.find('#NOUN') != -1:
            sentence = sentence.replace('#NOUN', str(__get_noun(counts)), 1)

            if sentence.find('#NOUN') == -1:
                return sentence

        return sentence
    else:
        return sentence
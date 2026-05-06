def __replace_verbs(sentence, counts):
    """Lets find and replace all instances of #VERB
    :param _sentence:
    :param counts:
    """

    if sentence is not None:
        while sentence.find('#VERB') != -1:
            sentence = sentence.replace('#VERB', str(__get_verb(counts)), 1)

            if sentence.find('#VERB') == -1:
                return sentence
        return sentence
    else:
        return sentence
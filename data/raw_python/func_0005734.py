def __replace_adjective(sentence, counts):
    """Lets find and replace all instances of #ADJECTIVE
    :param _sentence:
    :param counts:
    """

    if sentence is not None:

        while sentence.find('#ADJECTIVE') != -1:
            sentence = sentence.replace('#ADJECTIVE',
                                          str(__get_adjective(counts)), 1)

            if sentence.find('#ADJECTIVE') == -1:
                return sentence
        return sentence
    else:
        return sentence
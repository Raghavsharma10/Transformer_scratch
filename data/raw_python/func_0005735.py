def __replace_names(sentence, counts):
    """Lets find and replace all instances of #NAME
    :param _sentence:
    :param counts:
    """

    if sentence is not None:

        while sentence.find('#NAME') != -1:
            sentence = sentence.replace('#NAME', str(__get_name(counts)), 1)

            if sentence.find('#NAME') == -1:
                return sentence
        return sentence
    else:
        return sentence
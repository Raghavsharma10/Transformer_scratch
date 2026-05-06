def ___replace_adjective_maybe(sentence, counts):
    """Lets find and replace all instances of #ADJECTIVE_MAYBE
    :param _sentence:
    :param counts:
    """

    random_decision = random.randint(0, 1)

    if sentence is not None:

        while sentence.find('#ADJECTIVE_MAYBE') != -1:

            if random_decision % 2 == 0:
                sentence = sentence.replace('#ADJECTIVE_MAYBE',
                                              ' ' + str(__get_adjective(counts)), 1)
            elif random_decision % 2 != 0:
                sentence = sentence.replace('#ADJECTIVE_MAYBE', '', 1)

            if sentence.find('#ADJECTIVE_MAYBE') == -1:
                return sentence
        return sentence
    else:
        return sentence
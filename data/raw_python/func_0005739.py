def __replace_capitalise(sentence):
    """here we replace all instances of #CAPITALISE and cap the next word.
    ############

    #NOTE:  Buggy as hell, as it doesn't account for words that are already
    #capitalized
    ############

    :param _sentence:
    """

    if sentence is not None:
        while sentence.find('#CAPITALISE') != -1:

            cap_index = _sentence.find('#CAPITALISE')
            part1 = sentence[:cap_index]
            part2 = sentence[cap_index + 12:cap_index + 13]
            part3 = sentence[cap_index + 13:]

            if part2 in "abcdefghijklmnopqrstuvwxyz":
                sentence = part1 + part2.capitalize() + part3
            else:
                sentence = part1 + part2 + part3

        if sentence.find('#CAPITALISE') == -1:
            return sentence
    else:
        return sentence
def __replace_capall(sentence):
    """here we replace all instances of #CAPALL and cap the entire sentence.
    Don't believe that CAPALL is buggy anymore as it forces all uppercase OK?

    :param _sentence:
        """

    # print "\nReplacing CAPITALISE:  "

    if sentence is not None:
        while sentence.find('#CAPALL') != -1:
            # _cap_index = _sentence.find('#CAPALL')
            sentence = sentence.upper()
            sentence = sentence.replace('#CAPALL ', '', 1)

        if sentence.find('#CAPALL') == -1:
            return sentence
    else:
        return sentence
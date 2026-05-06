def __process_sentence(sentence_tuple, counts):
    """pull the actual sentence from the tuple (tuple contains additional data such as ID)
    :param _sentence_tuple:
    :param counts:
    """

    sentence = sentence_tuple[2]

    # now we start replacing words one type at a time...
    sentence = __replace_verbs(sentence, counts)

    sentence = __replace_nouns(sentence, counts)

    sentence = ___replace_adjective_maybe(sentence, counts)

    sentence = __replace_adjective(sentence, counts)

    sentence = __replace_names(sentence, counts)

    # here we perform a check to see if we need to use A or AN depending on the 
    # first letter of the following word...
    sentence = __replace_an(sentence)

    # replace the new repeating segments
    sentence = __replace_repeat(sentence)

    # now we will read, choose and substitute each of the RANDOM sentence tuples
    sentence = __replace_random(sentence)

    # now we are going to choose whether to capitalize words/sentences or not
    sentence = __replace_capitalise(sentence)

    # here we will choose whether to capitalize all words in the sentence
    sentence = __replace_capall(sentence)

    # check for appropriate spaces in the correct places.
    sentence = __check_spaces(sentence)

    return sentence
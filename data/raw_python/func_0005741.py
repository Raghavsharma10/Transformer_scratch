def __check_spaces(sentence):
    """
    Here we check to see that we have the correct number of spaces in the correct locations.

    :param _sentence:
    :return:
    """
    # We have to run the process multiple times:
    #   Once to search for all spaces, and check if there are adjoining spaces;
    #   The second time to check for 2 spaces after sentence-ending characters such as . and ! and ?

    if sentence is not None:

        words = sentence.split()

        new_sentence = ''

        for (i, word) in enumerate(words):

            if word[-1] in set('.!?'):
                word += ' '
            new_word = ''.join(word)
            new_sentence += ' ' + new_word

        # remove any trailing whitespace
        new_sentence = new_sentence.strip()

    return new_sentence
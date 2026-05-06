def __replace_an(sentence):
    """Lets find and replace all instances of #AN
    This is a little different, as this depends on whether the next
    word starts with a vowel or a consonant.

    :param _sentence:
    """

    if sentence is not None:
        while sentence.find('#AN') != -1:
            an_index = sentence.find('#AN')

            if an_index > -1:
                an_index += 4

                if sentence[an_index] in 'aeiouAEIOU':
                    sentence = sentence.replace('#AN', str('an'), 1)
                else:
                    sentence = sentence.replace('#AN', str('a'), 1)

            if sentence.find('#AN') == -1:
                return sentence
        return sentence
    else:
        return sentence
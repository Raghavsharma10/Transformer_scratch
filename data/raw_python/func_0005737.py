def __replace_random(sentence):
    """Lets find and replace all instances of #RANDOM
    :param _sentence:
    """

    sub_list = None
    choice = None

    if sentence is not None:

        while sentence.find('#RANDOM') != -1:

            random_index = sentence.find('#RANDOM')
            start_index = sentence.find('#RANDOM') + 8
            end_index = sentence.find(']')

            if sentence.find('#RANDOM') is not None:
                sub_list = sentence[start_index:end_index].split(',')

                choice = random.randint(1, int(sub_list[0]))
                # _sub_list[_choice]

            to_be_replaced = sentence[random_index:end_index + 1]
            sentence = sentence.replace(to_be_replaced, sub_list[choice], 1)

            if sentence.find('#RANDOM') == -1:
                return sentence

        return sentence
    else:
        return sentence
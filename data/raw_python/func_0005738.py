def __replace_repeat(sentence):
    """
    Allows the use of repeating random-elements such as in the 'Ten green bottles' type sentences.

    :param sentence:
    """

    ######### USE SENTENCE_ID 47 for testing!

    repeat_dict = {}

    if sentence is not None:

        while sentence.find('#DEFINE_REPEAT') != -1:
            begin_index = sentence.find('#DEFINE_REPEAT')
            start_index = begin_index + 15
            end_index = sentence.find(']')

            if sentence.find('#DEFINE_REPEAT') is not None:
                sub_list = sentence[start_index:end_index].split(',')
                choice = sub_list[0]
                repeat_text = sub_list[1]
                repeat_dict[choice] = repeat_text
                sentence = sentence.replace(sentence[begin_index:end_index + 1], '', 1)

        while sentence.find('#REPEAT') != -1:
            if sentence.find('#REPEAT') is not None:
                repeat_begin_index = sentence.find('#REPEAT')
                repeat_start_index = repeat_begin_index + 8
                # by searching from repeat_index below we don't encounter dodgy bracket-matching errors.
                repeat_end_index = sentence.find(']', repeat_start_index)
                repeat_index = sentence[repeat_start_index:repeat_end_index]

                if repeat_index in repeat_dict:
                    sentence = sentence.replace(sentence[repeat_begin_index:repeat_end_index + 1],
                                                  str(repeat_dict[repeat_index]))

        if sentence.find('#REPEAT') == -1:
            return sentence
        return sentence
    else:
        return sentence
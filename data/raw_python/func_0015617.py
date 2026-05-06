def get_batch_input(sentences,
                    max_word_len,
                    word_dict,
                    char_dict,
                    word_unknown=1,
                    char_unknown=1,
                    word_ignore_case=False,
                    char_ignore_case=False):
    """Convert sentences to desired input tensors.

    :param sentences: A list of lists representing the input sentences.
    :param max_word_len: The maximum allowed length of word.
    :param word_dict: Map a word to an integer. (0 and 1 should be preserved)
    :param char_dict: Map a character to an integer. (0 and 1 should be preserved)
    :param word_unknown: An integer representing the unknown word.
    :param char_unknown: An integer representing the unknown character.
    :param word_ignore_case: Word will be transformed to lower case before mapping.
    :param char_ignore_case: Character will be transformed to lower case before mapping.

    :return word_embd_input, char_embd_input: The desired inputs.
    """
    sentence_num = len(sentences)

    max_sentence_len = max(map(len, sentences))

    word_embd_input = [[0] * max_sentence_len for _ in range(sentence_num)]
    char_embd_input = [[[0] * max_word_len for _ in range(max_sentence_len)] for _ in range(sentence_num)]

    for sentence_index, sentence in enumerate(sentences):
        for word_index, word in enumerate(sentence):
            if word_ignore_case:
                word_key = word.lower()
            else:
                word_key = word
            word_embd_input[sentence_index][word_index] = word_dict.get(word_key, word_unknown)
            for char_index, char in enumerate(word):
                if char_index >= max_word_len:
                    break
                if char_ignore_case:
                    char = char.lower()
                char_embd_input[sentence_index][word_index][char_index] = char_dict.get(char, char_unknown)

    return [numpy.asarray(word_embd_input), numpy.asarray(char_embd_input)]
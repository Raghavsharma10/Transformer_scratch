def make_sentence(list_words):
        """
        Return a sentence from list of words.

        :param list list_words: list of words
        :returns: sentence
        :rtype: str
        """
        lw_len = len(list_words)

        if lw_len > 6:
            list_words.insert(lw_len // 2 + random.choice(range(-2, 2)), ',')

        sentence = ' '.join(list_words).replace(' ,', ',')

        return sentence.capitalize() + '.'
def count_token_occurrences(cls, words):
        """
        Creates a key/value set of word/count for a given sample of text

        :param words: full list of all tokens, non-unique
        :type words: list
        :return: key/value pairs of words and their counts in the list
        :rtype: dict
        """
        counts = {}
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        return counts
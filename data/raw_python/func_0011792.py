def get_sentences_list(self, sentences=1):
        """
        Return sentences in list.

        :param int sentences: how many sentences
        :returns: list of strings with sentence
        :rtype: list
        """
        if sentences < 1:
            raise ValueError('Param "sentences" must be greater than 0.')

        sentences_list = []

        while sentences:
            num_rand_words = random.randint(self.MIN_WORDS, self.MAX_WORDS)

            random_sentence = self.make_sentence(
                random.sample(self.words, num_rand_words))

            sentences_list.append(random_sentence)
            sentences -= 1

        return sentences_list
def random_words(self, letter=None, count=1):
        """
        Returns list of random words.

        :param str letter: letter
        :param int count: how much words
        :rtype: list
        :returns: list of random words
        :raises: ValueError
        """
        self.check_count(count)

        words = []

        if letter is None:
            all_words = list(
                chain.from_iterable(self.nouns.values()))

            try:
                words = sample(all_words, count)
            except ValueError:
                len_sample = len(all_words)
                raise ValueError('Param "count" must be less than {0}. \
(It is only {0} words)'.format(len_sample + 1, letter))

        elif type(letter) is not str:
            raise ValueError('Param "letter" must be string.')

        elif letter not in self.available_letters:
            raise ValueError(
                'Param "letter" must be in {0}.'.format(
                    self.available_letters))

        elif letter in self.available_letters:
            try:
                words = sample(self.nouns[letter], count)
            except ValueError:
                len_sample = len(self.nouns[letter])
                raise ValueError('Param "count" must be less than {0}. \
(It is only {0} words for letter "{1}")'.format(len_sample + 1, letter))

        return words
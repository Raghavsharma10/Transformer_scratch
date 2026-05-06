def random_nicks(self, letter=None, gender='u', count=1):
        """
        Return list of random nicks.

        :param str letter: letter
        :param str gender: ``'f'`` for female, ``'m'`` for male and None for both
        :param int count: how much nicks
        :rtype: list
        :returns: list of random nicks
        :raises: ValueError
        """
        self.check_count(count)

        nicks = []

        if gender not in ('f', 'm', 'u'):
            raise ValueError('Param "gender" must be in (f, m, u)')

        if letter is None:

            all_nicks = list(
                chain.from_iterable(self.nicknames[gender].values()))

            try:
                nicks = sample(all_nicks, count)
            except ValueError:
                len_sample = len(all_nicks)
                raise ValueError('Param "count" must be less than {0}. \
(It is only {0} words.")'.format(len_sample + 1))

        elif type(letter) is not str:
            raise ValueError('Param "letter" must be string.')

        elif letter not in self.available_letters:
            raise ValueError(
                'Param "letter" must be in "{0}".'.format(
                    self.available_letters))

        elif letter in self.available_letters:
            try:
                nicks = sample(self.nicknames[gender][letter], count)
            except ValueError:
                len_sample = len(self.nicknames[gender][letter])
                raise ValueError('Param "count" must be less than {0}. \
(It is only {0} nicks for letter "{1}")'.format(len_sample + 1, letter))

        return nicks
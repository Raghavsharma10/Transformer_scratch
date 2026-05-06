def new_common_diceware_password(self, number_of_words=6, hint=''):
        """
        Return a suggested password
        :param int number_of_words: number of words generated
        :param str hint:
        :return tuple: a suggested password and a sentence

        >>> GeneratePassword().new_common_diceware_password()
        ('rive2sidelauraarchitectss!mplytheOreticalassessMeNt$', [('Mynheer', False), (',', False), ('Sir', False), ('Francis', False), (',', False), ('the', False), ('riverside', True), ('laura', True), (',', False), ('the', False), ('very', False), ('architects', True), ('of', False), ('the', False), ('river', False), ('on', False), ('his', False), ('right', False), ('purling', False), ('simply', True), ('to', False), ('the', False), ('bay', False), ('past', False), ('fish', False), ('weirs', False), ('and', False), ('rocks', False), (',', False), ('and', False), ('ahead', False), ('the', False), ('theoretical', True), ('assessments', True)])
        """
        keywords = [self.wordlist.get_random_word() for _ in range(number_of_words)]
        password = self.conformizer.conformize(keywords)
        if hint:
            keywords = [hint] + keywords
        return password, self.to_sentence.from_keywords(keywords)
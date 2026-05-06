def haikunate(self, delimiter='-', token_length=4, token_hex=False, token_chars='0123456789'):
        """
        Generate heroku-like random names to use in your python applications

        :param delimiter: Delimiter
        :param token_length: TokenLength
        :param token_hex: TokenHex
        :param token_chars: TokenChars
        :type delimiter: str
        :type token_length: int
        :type token_hex: bool
        :type token_chars: str
        :return: heroku-like random string
        :rtype: str
        """
        if token_hex:
            token_chars = '0123456789abcdef'

        adjective = self._random_element(self._adjectives)
        noun = self._random_element(self._nouns)
        token = ''.join(self._random_element(token_chars) for _ in range(token_length))

        sections = [adjective, noun, token]
        return delimiter.join(filter(None, sections))
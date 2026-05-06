def new_pin(self, min_length=4, min_common=1000, timeout=20, refresh_timeout=3):
        """
        Return a suggested PIN

        :param int min_length: minimum length of the PIN generated
        :param int min_common: the minimal commonness to be considered convertible to a PIN
        :param float timeout: main timeout in seconds
        :param float refresh_timeout: timeout to new sentence
        :return str: a string of digits

        >>> GeneratePassword().new_pin()
        ('32700', [('His', False), ('mouth', True), ('was', False), ('open', False), (',', False), ('his', False), ('neck', True), ('corded', True), ('with', False), ('the', False), ('strain', True), ('of', False), ('his', False), ('screams', True)])
        """
        self.refresh(count_common=min_length, min_common=min_common, timeout=refresh_timeout)
        rating = self.sentence_tool.rate(self.tokens)

        start = time()
        while time() - start < timeout:
            pin = ''
            for token, commonness in rating:
                if commonness >= min_common:
                    key = self.mnemonic.word_to_key('major_system', token.lower())
                    if key is not None:
                        pin += key

            if len(pin) < min_length:
                self.refresh(count_common=min_length, min_common=min_common, timeout=refresh_timeout)
                rating = self.sentence_tool.rate(self.tokens)
            else:
                return pin, list(self.overlap_pin(pin, self.tokens))

        return None
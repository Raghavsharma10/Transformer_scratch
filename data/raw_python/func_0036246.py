def from_pin(self, pin, timeout=5):
        """
        Generate a sentence from PIN

        :param str pin: a string of digits
        :param float timeout: total time in seconds
        :return dict: {
            'sentence': sentence corresponding to the PIN,
            'overlap': overlapping positions, starting for 0
        }

        >>> ToSentence().from_pin('3492')
        [("Helva's", False), ('masking', True), ('was', False), ('not', False), ('without', False), ('real', True), (',', False), ('pretty', True), ('novels', True)]
        """
        return self.keyword_parse.from_initials_list([self.mnemonic.reality_to_starter('major_system', number)
                                                      for number in pin],
                                                     timeout)
def convert_to_american_phonetic_alphabet(self, arpabet):
        '''
        转换成美音
        :param arpabet:
        :return:
        '''

        word = self._convert_to_word(arpabet=arpabet)

        if not word:
            return None

        return word.translate_to_american_phonetic_alphabet()
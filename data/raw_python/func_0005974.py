def convert_to_english_phonetic_alphabet(self, arpabet):
        '''
        转换成英音
        :param arpabet:
        :return:
        '''

        word = self._convert_to_word(arpabet=arpabet)

        if not word:
            return None

        return word.translate_to_english_phonetic_alphabet()
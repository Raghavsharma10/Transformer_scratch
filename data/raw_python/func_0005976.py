def translate_to_american_phonetic_alphabet(self, hide_stress_mark=False):
        '''
        转换成美音音。只要一个元音的时候需要隐藏重音标识
        :param hide_stress_mark:
        :return:

        '''

        translations = self.stress.mark_ipa() if (not hide_stress_mark) and self.have_vowel else ""

        for phoneme in self._phoneme_list:
            translations += phoneme.american

        return translations
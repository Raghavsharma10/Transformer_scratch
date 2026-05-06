def translate_to_arpabet(self):
        '''
        转换成arpabet
        :return:
        '''

        translations = []

        for phoneme in self._phoneme_list:
            if phoneme.is_vowel:
                translations.append(phoneme.arpabet + self.stress.mark_arpabet())
            else:
                translations.append(phoneme.arpabet)

        return " ".join(translations)
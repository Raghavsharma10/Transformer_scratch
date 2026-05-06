def _char_ro_vowel(self, char_info, type):
        '''
        Returns the vowel part of a character in rōmaji.
        '''
        if type == CV:
            return char_info[3]

        if type == VOWEL or type == XVOWEL:
            return char_info[0]

        return None
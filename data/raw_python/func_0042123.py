def _set_vowel(self, vowel):
        '''
        Sets the currently active vowel, e.g. ア.

        Vowels act slightly differently from other characters. If one
        succeeds the same vowel (or consonant-vowel pair with the same vowel)
        then it acts like a long vowel marker. E.g. おねえ becomes onē.

        Hence, either we increment the long vowel marker count, or we
        flush the current character and set the active character to this.

        In some cases, the ウ becomes a consonant-vowel if it's
        paired with a small vowel. We will not know this until we see
        what comes after the ウ, so there's some backtracking
        if that's the case.
        '''
        vowel_info = kana_lt[vowel]
        vowel_ro = self.active_vowel_ro

        if self._is_long_vowel(vowel_ro, vowel_info[0]):
            # Check to see if the current vowel is ウ. If so,
            # we might need to backtrack later on in case the 'u'
            # turns into 'w' when ウ is coupled with a small vowel.
            if vowel_ro == 'u':
                self.has_u_lvm = True

            self._inc_lvmarker()
        else:
            # Not the same, so flush the active character and continue.
            self._set_char(vowel, VOWEL)

        self.active_vowel_info = vowel_info
        self.active_vowel = vowel
def _promote_solitary_xvowel(self):
        '''
        "Promotes" the current xvowel to a regular vowel, in case
        it is not otherwise connected to a character.
        Used to print small vowels that would otherwise get lost;
        normally small vowels always form a pair, but in case one is
        by itself it should basically act like a regular vowel.
        '''
        char_type = self.active_char_type

        # Only promote if we actually have an xvowel, and if the currently
        # active character is not a consonant-vowel pair or vowel.
        if char_type == VOWEL or char_type == CV or self.active_xvowel is None:
            return

        self._set_char(self.active_xvowel, XVOWEL)
        self.active_xvowel = None
        self.active_xvowel_info = None
def _set_char(self, char, type):
        '''
        Sets the currently active character, e.g. ト. We save some information
        about the character as well. active_char_info contains the full
        tuple of rōmaji info, and active_ro_vowel contains e.g. 'o' for ト.

        We also set the character type: either a consonant-vowel pair
        or a vowel. This affects the way the character is flushed later.
        '''
        self.next_char_info = self._char_lookup(char)
        self.next_char_type = type
        self._flush_char()

        self.active_char = char
        self.active_char_type = type

        self.active_char_info = self._char_lookup(char)
        self.active_vowel_ro = self._char_ro_vowel(self.active_char_info, type)
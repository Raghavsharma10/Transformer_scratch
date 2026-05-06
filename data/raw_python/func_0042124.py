def _set_xvowel(self, xvowel):
        '''
        Sets the currently active small vowel, e.g. ァ.

        If an active small vowel has already been set, the current character
        must be flushed. (Double small vowels don't occur in dictionary
        words.) After that, we'll set the current character to this small
        vowel; in essence, it will act like a regular size vowel.

        We'll check for digraphs too, just so e.g. しょ followed by ぉ acts
        like a long vowel marker. This doesn't occur in dictionary words,
        but it's the most sensible behavior for unusual input.

        If the currently active character ends with the same vowel as this
        small vowel, a long vowel marker is added instead.
        E.g. テェ becomes 'tē'.
        '''
        xvowel_info = kana_lt[xvowel]
        vowel_info = self.active_vowel_info
        dgr_b_info = None

        # Special case: if the currently active character is 'n', we must
        # flush the character and set this small vowel as the active character.
        # This is because small vowels cannot affect 'n' like regular
        # consonant-vowel pairs.
        curr_is_n = self.active_vowel_ro == 'n'

        # Special case: if we've got an active vowel with special cases
        # attached to it (only ウ), and the small vowel that follows it
        # activates that special case, we may need to backtrack a bit.
        # This is because ウ is normally 'u' but becomes 'w' if there's
        # a small vowel right behind it (except the small 'u').
        # The 'w' behaves totally different from a standard vowel.
        if self.has_u_lvm and \
           xvowel_info is not None and \
           vowel_info is not None and \
           len(vowel_info) > 2 and \
           vowel_info[2].get('xv') is not None and \
           vowel_info[2]['xv'].get(xvowel_info[0]) is not None:
            # Decrement the long vowel marker, which was added on the
            # assumption that the 'u' is a vowel.
            self._dec_lvmarker()
            # Save the current vowel. We'll flush the current character,
            # without this vowel, and then set it again from a clean slate.
            former_vowel = self.active_vowel
            self.active_vowel_info = None
            self._flush_char()
            self._set_char(former_vowel, VOWEL)

        if self.active_vowel_ro == xvowel_info[0]:
            # We have an active character whose vowel is the same.
            self._inc_lvmarker()
        elif self.has_xvowel is True:
            # We have an active small vowel already. Flush the current
            # character and act as though the current small vowel
            # is a regular vowel.
            self._flush_char()
            self._set_char(xvowel, XVOWEL)
            return
        elif self.has_digraph_b is True:
            # We have an active digraph (two parts).
            dgr_b_info = self.active_dgr_b_info

        if curr_is_n:
            self._set_char(xvowel, XVOWEL)
            return

        if dgr_b_info is not None:
            if self._is_long_vowel(self.active_vowel_ro, dgr_b_info[0]) or \
               self._is_long_vowel(self.active_dgr_b_info[0], dgr_b_info[0]):
                # Same vowel as the one that's currently active.
                self._inc_lvmarker()
            else:
                # Not the same, so flush the active character and continue.
                self.active_vowel_ro = self.active_xvowel_info[0]
                self._set_char(xvowel, XVOWEL)
        else:
            self.active_xvowel = xvowel
            self.active_xvowel_info = xvowel_info

        self.has_xvowel = True
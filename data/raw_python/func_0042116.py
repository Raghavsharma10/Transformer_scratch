def _flush_char(self):
        '''
        Appends the rōmaji characters that represent the current state
        of the machine. For example, if the state includes the character
        ト, plus a geminate marker and a long vowel marker, this causes
        the characters "ttō" to be added to the output.
        '''
        # Ignore in case there's no active character, only at the
        # first iteration of the conversion process.
        if self.active_char is None:
            if self.unknown_char is not None:
                self._append_unknown_char()

            return

        char_info = self.active_char_info
        char_type = self.active_char_type
        char_ro = char_info[0]
        xv = self.active_xvowel_info
        di_b = self.active_dgr_b_info
        gem = self.geminate_count
        lvm = self.lvmarker_count

        # Check for special combinations. This is exceptional behavior
        # for very specific character combinations, too unique to
        # build into the data model for every kana.
        # If a special combination is found, we'll replace the
        # rōmaji character we were planning on flushing.
        if char_type == VOWEL and len(char_info) >= 3 and xv is not None:
            try:
                exc = char_info[2]['xv'][xv[0]]
                # Found a special combination. Replace the rōmaji character.
                char_ro = exc
            except (IndexError, KeyError):
                # IndexError: no 'xv' exceptions list for this vowel.
                # KeyError: no exception for the current small vowel.
                pass

        # Check whether we're dealing with a valid char type.
        if char_type not in CHAR_TYPES:
            raise InvalidCharacterTypeError

        # If no modifiers are active (geminate marker, small vowel marker,
        # etc.) then just the currently active character is flushed.
        # We'll also continue if the character is 'n', which has a special
        # case attached to it that we'll tackle down below.
        if xv is di_b is None and gem == lvm == 0 and char_ro != 'n':
            self._append_to_stack(char_ro)
            self._append_unknown_char()
            self._clear_char()
            return

        # At this point, we're considering two main factors: the currently
        # active character, and possibly a small vowel character if one is set.
        # For example, if the active character is テ and a small vowel ィ
        # is set, the result is 'ti'. If no small vowel is set, just
        # plain 'te' comes out.
        #
        # Aside from this choice, we're also considering the number of active
        # long vowel markers, which repeats the vowel part. If there's
        # at least one long vowel marker, we also use a macron vowel
        # rather than the regular one, e.g. 'ī' instead of 'i'.

        if char_type == CV:
            # Deconstruct the info object for clarity.
            char_gem_cons = char_info[1]  # the extra geminate consonant
            char_cons = char_info[2]      # the consonant part
            char_lv = char_info[4]        # the long vowel part

            # If this flushed character is an 'n', and precedes a vowel or
            # a 'y' consonant, it must be followed by an apostrophe.
            char_apostrophe = ''

            if char_ro == 'n' and self.next_char_info is not None:
                first_char = None

                if self.next_char_type == CV:
                    first_char = self._char_ro_cons(
                        self.next_char_info,
                        CV
                    )

                if self.next_char_type == VOWEL or \
                   self.next_char_type == XVOWEL:
                    first_char = self._char_ro_vowel(
                        self.next_char_info,
                        VOWEL
                    )

                # If the following character is in the set of characters
                # that should trigger an apostrophe, add it to the output.
                if first_char in n_apostrophe:
                    char_apostrophe = APOSTROPHE_CHAR

            # Check to see if we've got a full digraph.
            if self.active_dgr_a_info is not None and \
               self.active_dgr_b_info is not None:
                char_cons = self.active_dgr_a_info[0]

            # Determine the geminate consonant part (which can be
            # arbitrarily long).
            gem_cons = char_gem_cons * gem

            if xv is not None:
                # Combine the consonant of the character with the small vowel.
                # Use a macron vowel if there's a long vowel marker,
                # else use the regular vowel.
                vowel = xv[1] * lvm if lvm > 0 else xv[0]
            elif di_b is not None:
                # Put together the digraph. Here we produce the latter half
                # of the digraph.
                vowel = di_b[1] * lvm if lvm > 0 else di_b[0]
            else:
                # Neither a small vowel marker, nor a digraph.
                vowel = ''

            if vowel != '':
                # If we've got a special vowel part, combine it with the
                # main consonant.
                char_main = char_cons + char_apostrophe + vowel
            else:
                # If not, process the main character and add the long vowels
                # if applicable.
                if lvm > 0:
                    char_main = char_cons + char_apostrophe + char_lv * lvm
                else:
                    char_main = char_ro + char_apostrophe

            self._append_to_stack(gem_cons + char_main)

        if char_type == VOWEL or char_type == XVOWEL:
            char_lv = char_info[1]  # the long vowel part

            if xv is not None:
                xv_ro = xv[1] * lvm if lvm > 0 else xv[0]
                self._append_to_stack(char_ro + xv_ro)
            else:
                vowel_ro = char_lv * lvm if lvm > 0 else char_ro
                self._append_to_stack(vowel_ro)

        # Append unknown the character as well.
        self._append_unknown_char()
        self._clear_char()
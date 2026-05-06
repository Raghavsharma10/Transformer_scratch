def _add_unknown_char(self, string):
        '''
        Adds an unknown character to the stack.
        '''
        if self.has_xvowel:
            # Ensure an xvowel gets printed if we've got an active
            # one right now.
            self._promote_solitary_xvowel()

        self.unknown_char = string
        self._flush_char()
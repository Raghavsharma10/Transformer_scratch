def _clear_char(self):
        '''
        Clears the current character and makes the machine ready
        to accept the next character.
        '''
        self.lvmarker_count = 0
        self.geminate_count = 0
        self.next_char_info = None
        self.next_char_type = None
        self.active_vowel = None
        self.active_vowel_info = None
        self.active_vowel_ro = None
        self.active_xvowel = None
        self.active_xvowel_info = None
        self.active_char = None
        self.active_char_info = None
        self.active_char_type = None
        self.active_dgr_a_info = None
        self.active_dgr_b_info = None
        self.has_xvowel = False
        self.has_digraph_b = False
        self.has_u_lvm = False
        self.unknown_char = None
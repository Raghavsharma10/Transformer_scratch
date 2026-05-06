def _set_digraph_b(self, char):
        '''
        Sets the second part of a digraph.
        '''
        self.has_digraph_b = True
        # Change the active vowel to the one provided by the second part
        # of the digraph.
        self.active_vowel_ro = di_b_lt[char][0]
        self.active_dgr_b_info = di_b_lt[char]
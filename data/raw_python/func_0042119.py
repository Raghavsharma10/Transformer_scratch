def _set_digraph_a(self, char):
        '''
        Sets the currently active character, in case it is (potentially)
        the first part of a digraph.
        '''
        self._set_char(char, CV)
        self.active_dgr_a_info = di_a_lt[char]
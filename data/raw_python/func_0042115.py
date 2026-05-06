def _append_unknown_char(self):
        '''
        Appends the unknown character, in case one was encountered.
        '''
        if self.unknown_strategy == UNKNOWN_INCLUDE and \
           self.unknown_char is not None:
            self._append_to_stack(self.unknown_char)

        self.unknown_char = None
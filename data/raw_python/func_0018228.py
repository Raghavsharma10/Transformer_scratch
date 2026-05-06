def ansi_to_curses(self, char):
        '''Handles reading ANSI escape sequences'''
        # ANSI sequences are:
        # ESC [ <key>
        # If we see ESC, read a char
        if char != ESC:
            return char
        # If we see [, read another char
        if self.getc(block=True) != ANSI_START_SEQ:
            self._readline_echo(BELL, True)
            return theNULL
        key = self.getc(block=True)
        # Translate the key to curses
        try:
            return ANSI_KEY_TO_CURSES[key]
        except:
            self._readline_echo(BELL, True)
            return theNULL
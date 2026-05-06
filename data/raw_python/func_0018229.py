def readline(self, echo=None, prompt='', use_history=True):
        """Return a line of text, including the terminating LF
           If echo is true always echo, if echo is false never echo
           If echo is None follow the negotiated setting.
           prompt is the current prompt to write (and rewrite if needed)
           use_history controls if this current line uses (and adds to) the command history.
        """
        
        line = []
        insptr = 0
        ansi = 0
        histptr = len(self.history)
            
        if self.DOECHO:
            self.write(prompt)
            self._current_prompt = prompt
        else:
            self._current_prompt = ''
        
        self._current_line = ''
        
        while True:
            c = self.getc(block=True)
            c = self.ansi_to_curses(c)
            if c == theNULL:
                continue
            
            elif c == curses.KEY_LEFT:
                if insptr > 0:
                    insptr = insptr - 1
                    self._readline_echo(self.CODES['CSRLEFT'], echo)
                else:
                    self._readline_echo(BELL, echo)
                continue
            elif c == curses.KEY_RIGHT:
                if insptr < len(line):
                    insptr = insptr + 1
                    self._readline_echo(self.CODES['CSRRIGHT'], echo)
                else:
                    self._readline_echo(BELL, echo)
                continue
            elif c == curses.KEY_UP or c == curses.KEY_DOWN:
                if not use_history:
                    self._readline_echo(BELL, echo)
                    continue
                if c == curses.KEY_UP:
                    if histptr > 0:
                        histptr = histptr - 1
                    else:
                        self._readline_echo(BELL, echo)
                        continue
                elif c == curses.KEY_DOWN:
                    if histptr < len(self.history):
                        histptr = histptr + 1
                    else:
                        self._readline_echo(BELL, echo)
                        continue
                line = []
                if histptr < len(self.history):
                    line.extend(self.history[histptr])
                for char in range(insptr):
                    self._readline_echo(self.CODES['CSRLEFT'], echo)
                self._readline_echo(self.CODES['DEOL'], echo)
                self._readline_echo(''.join(line), echo)
                insptr = len(line)
                continue
            elif c == chr(3):
                self._readline_echo('\n' + curses.ascii.unctrl(c) + ' ABORT\n', echo)
                return ''
            elif c == chr(4):
                if len(line) > 0:
                    self._readline_echo('\n' + curses.ascii.unctrl(c) + ' ABORT (QUIT)\n', echo)
                    return ''
                self._readline_echo('\n' + curses.ascii.unctrl(c) + ' QUIT\n', echo)
                return 'QUIT'
            elif c == chr(10):
                self._readline_echo(c, echo)
                result = ''.join(line)
                if use_history:
                    self.history.append(result)
                if echo is False:
                    if prompt:
                        self.write( chr(10) )
                    log.debug('readline: %s(hidden text)', prompt)
                else:
                    log.debug('readline: %s%r', prompt, result)
                return result
            elif c == curses.KEY_BACKSPACE or c == chr(127) or c == chr(8):
                if insptr > 0:
                    self._readline_echo(self.CODES['CSRLEFT'] + self.CODES['DEL'], echo)
                    insptr = insptr - 1
                    del line[insptr]
                else:
                    self._readline_echo(BELL, echo)
                continue
            elif c == curses.KEY_DC:
                if insptr < len(line):
                    self._readline_echo(self.CODES['DEL'], echo)
                    del line[insptr]
                else:
                    self._readline_echo(BELL, echo)
                continue
            else:
                if ord(c) < 32:
                    c = curses.ascii.unctrl(c)
                if len(line) > insptr:
                    self._readline_insert(c, echo, insptr, line)
                else:
                    self._readline_echo(c, echo)
            line[insptr:insptr] = c
            insptr = insptr + len(c)
            if self._readline_do_echo(echo):
                self._current_line = line
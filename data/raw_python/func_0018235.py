def inputcooker(self):
        """Input Cooker - Transfer from raw queue to cooked queue.

        Set self.eof when connection is closed.  Don't block unless in
        the midst of an IAC sequence.
        """
        try:
            while True:
                c = self._inputcooker_getc()
                if not self.iacseq:
                    if c == IAC:
                        self.iacseq += c
                        continue
                    elif c == chr(13) and not(self.sb):
                        c2 = self._inputcooker_getc(block=False)
                        if c2 == theNULL or c2 == '':
                            c = chr(10)
                        elif c2 == chr(10):
                            c = c2
                        else:
                            self._inputcooker_ungetc(c2)
                            c = chr(10)
                    elif c in [x[0] for x in self.ESCSEQ.keys()]:
                        'Looks like the begining of a key sequence'
                        codes = c
                        for keyseq in self.ESCSEQ.keys():
                            if len(keyseq) == 0:
                                continue
                            while codes == keyseq[:len(codes)] and len(codes) <= keyseq:
                                if codes == keyseq:
                                    c = self.ESCSEQ[keyseq]
                                    break
                                codes = codes + self._inputcooker_getc()
                            if codes == keyseq:
                                break
                            self._inputcooker_ungetc(codes[1:])
                            codes = codes[0]
                    self._inputcooker_store(c)
                elif len(self.iacseq) == 1:
                    'IAC: IAC CMD [OPTION only for WILL/WONT/DO/DONT]'
                    if c in (DO, DONT, WILL, WONT):
                        self.iacseq += c
                        continue
                    self.iacseq = ''
                    if c == IAC:
                        self._inputcooker_store(c)
                    else:
                        if c == SB: # SB ... SE start.
                            self.sb = 1
                            self.sbdataq = ''
                        elif c == SE: # SB ... SE end.
                            self.sb = 0
                        # Callback is supposed to look into
                        # the sbdataq
                        self.options_handler(self.sock, c, NOOPT)
                elif len(self.iacseq) == 2:
                    cmd = self.iacseq[1]
                    self.iacseq = ''
                    if cmd in (DO, DONT, WILL, WONT):
                        self.options_handler(self.sock, cmd, c)
        except (EOFError, socket.error):
            pass
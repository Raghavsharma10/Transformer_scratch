def setterm(self, term):
        "Set the curses structures for this terminal"
        log.debug("Setting termtype to %s" % (term, ))
        curses.setupterm(term) # This will raise if the termtype is not supported
        self.TERM = term
        self.ESCSEQ = {}
        for k in self.KEYS.keys():
            str = curses.tigetstr(curses.has_key._capability_names[k])
            if str:
                self.ESCSEQ[str] = k
        # Create a copy to prevent altering the class
        self.CODES = self.CODES.copy()
        self.CODES['DEOL'] = curses.tigetstr('el')
        self.CODES['DEL'] = curses.tigetstr('dch1')
        self.CODES['INS'] = curses.tigetstr('ich1')
        self.CODES['CSRLEFT'] = curses.tigetstr('cub1')
        self.CODES['CSRRIGHT'] = curses.tigetstr('cuf1')
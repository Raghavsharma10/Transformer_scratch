def scan_upto(self, regex):
        """
        Scan up to, but not including, the given regex.

            >>> s = Scanner("test string")
            >>> s.scan('t')
            't'
            >>> s.scan_upto(r' ')
            'est'
            >>> s.pos
            4
            >>> s.pos_history
            [0, 1, 4]
        """
        pos = self.pos
        if self.scan_until(regex) is not None:
            self.pos -= len(self.matched())
            # Remove the intermediate position history entry.
            self.pos_history.pop(-2)
            return self.pre_match()[pos:]
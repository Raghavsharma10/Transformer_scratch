def unscan(self):
        """
        Undo the last scan, resetting the position and match registers.

            >>> s = Scanner('test string')
            >>> s.pos
            0
            >>> s.skip(r'te')
            2
            >>> s.rest
            'st string'
            >>> s.unscan()
            >>> s.pos
            0
            >>> s.rest
            'test string'
        """
        self.pos_history.pop()
        self._pos = self.pos_history[-1]
        self.match_history.pop()
        self._match = self.match_history[-1]
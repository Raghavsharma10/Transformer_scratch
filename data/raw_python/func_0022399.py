def process(self, chunk):
        """
        computes the hash of all of the trigrams in the chunk using a window
        of length 5
        """
        self._digest = None

        if isinstance(chunk, text_type):
            chunk = chunk.encode('utf-8')

        # chunk is a byte string
        for char in chunk:
            self.num_char += 1
            if PY3:
                # In Python 3, iterating over bytes yields integers
                c = char
            else:
                c = ord(char)
            if len(self.window) > 1:            # seen at least three characters
                self.acc[self.tran_hash(c, self.window[0], self.window[1], 0)] += 1
            if len(self.window) > 2:            # seen at least four characters
                self.acc[self.tran_hash(c, self.window[0], self.window[2], 1)] += 1
                self.acc[self.tran_hash(c, self.window[1], self.window[2], 2)] += 1
            if len(self.window) > 3:            # have a full window
                self.acc[self.tran_hash(c, self.window[0], self.window[3], 3)] += 1
                self.acc[self.tran_hash(c, self.window[1], self.window[3], 4)] += 1
                self.acc[self.tran_hash(c, self.window[2], self.window[3], 5)] += 1
                # duplicate hashes, used to maintain 8 trigrams per character
                self.acc[self.tran_hash(self.window[3], self.window[0], c, 6)] += 1
                self.acc[self.tran_hash(self.window[3], self.window[2], c, 7)] += 1

            # add current character to the window, remove the previous character
            if len(self.window) < 4:
                self.window = [c] + self.window
            else:
                self.window = [c] + self.window[:3]
def next(self, type=None):
        """ Returns the next chunk in the sentence with the given type.
        """
        i = self.stop
        s = self.sentence
        while i < len(s):
            if s[i].chunk is not None and type in (s[i].chunk.type, None):
                return s[i].chunk
            i += 1
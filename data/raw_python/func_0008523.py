def previous(self, type=None):
        """ Returns the next previous chunk in the sentence with the given type.
        """
        i = self.start - 1
        s = self.sentence
        while i > 0:
            if s[i].chunk is not None and type in (s[i].chunk.type, None):
                return s[i].chunk
            i -= 1
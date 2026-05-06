def next(self, type=None):
        """ Returns the next word in the sentence with the given type.
        """
        i = self.index + 1
        s = self.sentence
        while i < len(s):
            if type in (s[i].type, None):
                return s[i]
            i += 1
def previous(self, type=None):
        """ Returns the next previous word in the sentence with the given type.
        """
        i = self.index - 1
        s = self.sentence
        while i > 0:
            if type in (s[i].type, None):
                return s[i]
            i -= 1
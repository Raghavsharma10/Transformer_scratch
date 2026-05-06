def _string_to_int(self, s):
        """Read an integer in s, in Little Indian. """
        base = len(self.alphabet)
        return sum((self._letter_to_int(l) * base**lsb 
                    for lsb, l in enumerate(s)
                   ))
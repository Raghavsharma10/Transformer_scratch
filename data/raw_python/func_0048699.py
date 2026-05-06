def get(self, index1, index2=None):
        """Return the text from INDEX1 to INDEX2 (not included)."""
        returnvar = self.tk.call(self._w, 'get', index1, index2)
        return returnvar.strip()
def ok(self):
        """
        Returns True if OK to use, else False
        """
        try:
            v = float(self._value)
            if v < self.fmin or v > self.fmax:
                return False
            else:
                return True
        except:
            return False
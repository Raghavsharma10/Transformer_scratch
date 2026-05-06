def add(self, num):
        """
        Adds num to the current value
        """
        try:
            val = self.value() + num
        except:
            val = num
        self.set(min(self.fmax, max(self.fmin, val)))
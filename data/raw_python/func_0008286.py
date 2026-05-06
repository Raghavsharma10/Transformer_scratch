def sub(self, num):
        """
        Subtracts num from the current value
        """
        try:
            val = self.value() - num
        except:
            val = -num
        self.set(max(0, val))
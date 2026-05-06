def set(self, num):
        """
        Sets current value to num
        """
        if self.validate(num) is not None:
            self.index = self.allowed.index(num)
        IntegerEntry.set(self, num)
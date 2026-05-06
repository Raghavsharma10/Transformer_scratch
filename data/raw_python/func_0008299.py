def add(self, num):
        """
        Adds num to the current value
        """
        self.index = max(0, min(len(self.allowed)-1, self.index+num))
        self.set(self.allowed[self.index])
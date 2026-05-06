def add(self, quantity):
        """
        Adds an angle to the value
        """
        newvalue = self._value + quantity
        self.set(newvalue.deg)
def sub(self, quantity):
        """
        Subtracts an angle from the value
        """
        newvalue = self._value - quantity
        self.set(newvalue.deg)
def set(self, num):
        """
        Sets the current value equal to num
        """
        self._value = str(round(float(num), self.nplaces))
        self._variable.set(self._value)
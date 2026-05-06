def set(self, num):
        """
        Sets the current value equal to num
        """
        self._value = str(int(num))
        self._variable.set(self._value)
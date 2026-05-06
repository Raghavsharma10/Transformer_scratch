def set(self, num):
        """
        Sets the current value equal to num
        """
        self._value = coord.Angle(num, unit=u.deg)
        self._variable.set(self.as_string())
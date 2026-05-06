def ok(self):
        """
        Returns True if OK to use, else False
        """
        try:
            coord.Angle(self._value, unit=u.deg)
            return True
        except ValueError:
            return False
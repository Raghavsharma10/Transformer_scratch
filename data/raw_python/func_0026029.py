def checkValue(self,value,strict=0):
        """Check and convert a parameter value.

        Raises an exception if the value is not permitted for this
        parameter.  Otherwise returns the value (converted to the
        right type.)
        """
        v = self._coerceValue(value,strict)
        return self.checkOneValue(v,strict)
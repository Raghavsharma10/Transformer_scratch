def set(self, value, field=None, index=None, check=1):
        """Set value of this parameter from a string or other value.
        Field is optional parameter field (p_prompt, p_minimum, etc.)
        Index is optional array index (zero-based).  Set check=0 to
        assign the value without checking to see if it is within
        the min-max range or in the choice list."""

        if index is not None:
            raise SyntaxError("Parameter "+self.name+" is not an array")

        if field:
            self._setField(value,field,check=check)
        else:
            if check:
                self.value = self.checkValue(value)
            else:
                self.value = self._coerceValue(value)
            self.setChanged()
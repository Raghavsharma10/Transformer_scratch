def set(self, value, field=None, index=None, check=1):
        """Set value of this parameter from a string or other value.
        Field is optional parameter field (p_prompt, p_minimum, etc.)
        Index is optional array index (zero-based).  Set check=0 to
        assign the value without checking to see if it is within
        the min-max range or in the choice list."""
        if index is not None:
            sumindex = self._sumindex(index)
            try:
                value = self._coerceOneValue(value)
                if check:
                    self.value[sumindex] = self.checkOneValue(value)
                else:
                    self.value[sumindex] = value
                return
            except IndexError:
                # should never happen
                raise SyntaxError("Illegal index [" + repr(sumindex) +
                        "] for array parameter " + self.name)
        if field:
            self._setField(value,field,check=check)
        else:
            if check:
                self.value = self.checkValue(value)
            else:
                self.value = self._coerceValue(value)
            self.setChanged()
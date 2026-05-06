def get(self, field=None, index=None, lpar=0, prompt=1, native=0, mode="h"):
        """Return value of this parameter as a string (or in native format
        if native is non-zero.)"""

        if field and field != "p_value":
            # note p_value comes back to this routine, so shortcut that case
            return self._getField(field,native=native,prompt=prompt)

        # may prompt for value if prompt flag is set
        if prompt: self._optionalPrompt(mode)

        if index is not None:
            raise SyntaxError("Parameter "+self.name+" is not an array")

        if native:
            rv = self.value
        else:
            rv = self.toString(self.value)
        return rv
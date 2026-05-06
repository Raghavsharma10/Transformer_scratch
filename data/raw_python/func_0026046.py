def _coerceValue(self,value,strict=0):
        """Coerce parameter to appropriate type

        Should accept None or null string.  Must be an array.
        """
        try:
            if isinstance(value,str):
                # allow single blank-separated string as input
                value = value.split()
            if len(value) != len(self.value):
                raise IndexError
            v = len(self.value)*[0]
            for i in range(len(v)):
                v[i] = self._coerceOneValue(value[i],strict)
            return v
        except (IndexError, TypeError):
            raise ValueError("Value must be a " + repr(len(self.value)) +
                    "-element array for " + self.name)
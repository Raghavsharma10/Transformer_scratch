def _setField(self, value, field, check=1):
        """Set a parameter field value"""
        try:
            # expand field name using minimum match
            field = _setFieldDict[field]
        except KeyError as e:
            raise SyntaxError("Cannot set field " + field +
                    " for parameter " + self.name + "\n" + str(e))
        if field == "p_prompt":
            self.prompt = irafutils.removeEscapes(irafutils.stripQuotes(value))
        elif field == "p_value":
            self.set(value,check=check)
        elif field == "p_filename":
            # this is only relevant for list parameters (*imcur, *gcur, etc.)
            self.set(value,check=check)
        elif field == "p_scope":
            self.scope = value
        elif field == "p_maximum":
            self.max = self._coerceOneValue(value)
        elif field == "p_minimum":
            if isinstance(value,str) and '|' in value:
                self._setChoice(irafutils.stripQuotes(value))
            else:
                self.min = self._coerceOneValue(value)
        elif field == "p_mode":
            # not doing any type or value checking here -- setting mode is
            # rare, so assume that it is being done correctly
            self.mode = irafutils.stripQuotes(value)
        else:
            raise RuntimeError("Program bug in IrafPar._setField()" +
                    "Requested field " + field + " for parameter " + self.name)
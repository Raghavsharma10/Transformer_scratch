def get(self, field=None, index=None, lpar=0, prompt=1, native=0, mode="h"):
        """Return value of this parameter as a string (or in native format
        if native is non-zero.)"""

        if field: return self._getField(field,native=native,prompt=prompt)

        # may prompt for value if prompt flag is set
        #XXX should change _optionalPrompt so we prompt for each element of
        #XXX the array separately?  I think array parameters are
        #XXX not useful as non-hidden params.

        if prompt: self._optionalPrompt(mode)

        if index is not None:
            sumindex = self._sumindex(index)
            try:
                if native:
                    return self.value[sumindex]
                else:
                    return self.toString(self.value[sumindex])
            except IndexError:
                # should never happen
                raise SyntaxError("Illegal index [" + repr(sumindex) +
                        "] for array parameter " + self.name)
        elif native:
            # return object itself for an array because it is
            # indexable, can have values assigned, etc.
            return self
        else:
            # return blank-separated string of values for array
            return str(self)
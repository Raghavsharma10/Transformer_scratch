def _getField(self, field, native=0, prompt=1):
        """Get a parameter field value"""
        try:
            # expand field name using minimum match
            field = _getFieldDict[field]
        except KeyError as e:
            # re-raise the exception with a bit more info
            raise SyntaxError("Cannot get field " + field +
                    " for parameter " + self.name + "\n" + str(e))
        if field == "p_value":
            # return value of parameter
            # Note that IRAF returns the filename for list parameters
            # when p_value is used.  I consider this a bug, and it does
            # not appear to be used by any cl scripts or SPP programs
            # in either IRAF or STSDAS.  It is also in conflict with
            # the IRAF help documentation.  I am making p_value exactly
            # the same as just a simple CL parameter reference.
            return self.get(native=native,prompt=prompt)
        elif field == "p_name": return self.name
        elif field == "p_xtype": return self.type
        elif field == "p_type": return self._getPType()
        elif field == "p_mode": return self.mode
        elif field == "p_prompt": return self.prompt
        elif field == "p_scope": return self.scope
        elif field == "p_default" or field == "p_filename":
            # these all appear to be equivalent -- they just return the
            # current PFilename of the parameter (which is the same as the value
            # for non-list parameters, and is the filename for list parameters)
            return self._getPFilename(native,prompt)
        elif field == "p_maximum":
            if native:
                return self.max
            else:
                return self.toString(self.max)
        elif field == "p_minimum":
            if self.choice is not None:
                if native:
                    return self.choice
                else:
                    schoice = list(map(self.toString, self.choice))
                    return "|" + "|".join(schoice) + "|"
            else:
                if native:
                    return self.min
                else:
                    return self.toString(self.min)
        else:
            # XXX unimplemented fields:
            # p_length: maximum string length in bytes -- what to do with it?
            raise RuntimeError("Program bug in IrafPar._getField()\n" +
                    "Requested field " + field + " for parameter " + self.name)
def convertToNative(self, aVal):
        """ Convert to native bool; interpret certain strings. """
        if aVal is None:
            return None
        if isinstance(aVal, bool): return aVal
        # otherwise interpret strings
        return str(aVal).lower() in ('1','on','yes','true')
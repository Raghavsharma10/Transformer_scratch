def dpar(self, cl=1):
        """Return dpar-style executable assignment for parameter

        Default is to write CL version of code; if cl parameter is
        false, writes Python executable code instead.
        """
        sval = self.toString(self.value, quoted=1)
        if not cl:
            if sval == "": sval = "None"
        s = "%s = %s" % (self.name, sval)
        return s
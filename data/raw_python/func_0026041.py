def dpar(self, cl=1):
        """Return dpar-style executable assignment for parameter

        Default is to write CL version of code; if cl parameter is
        false, writes Python executable code instead.  Note that
        dpar doesn't even work for arrays in the CL, so we just use
        Python syntax here.
        """
        sval = list(map(self.toString, self.value, len(self.value)*[1]))
        for i in range(len(sval)):
            if sval[i] == "":
                sval[i] = "None"
        s = "%s = [%s]" % (self.name, ', '.join(sval))
        return s
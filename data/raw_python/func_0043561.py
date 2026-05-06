def provides(self, imt):
        """
        Returns True iff the self is at least as specific as other.

        Examples:
        application/xhtml+xml provides application/xml, application/*, */*
        text/html provides text/*, but not application/xhtml+xml or application/html
        """
        return self.type[:imt.specifity] == imt.type[:imt.specifity]
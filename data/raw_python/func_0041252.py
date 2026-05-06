def _validateAttrs(self, keys):
        """prove that all attributes are defined appropriately"""
        badAttrsMsg = ""
        for k in keys:
            if k not in self.attrs:
                badAttrsMsg += "Attribute key '%s' is not a valid attribute"%(k)
        if badAttrsMsg:
            raise ValueError("Encountered invalid attributes.  ALLOWED: %s%s%s"\
                %(list(self.attrs), os.linesep, badAttrsMsg))
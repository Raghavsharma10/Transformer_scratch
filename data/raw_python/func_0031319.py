def getbr(self, name):
        """ Return a bridge object."""
        for br in self.showall():
            if br.name == name:
                return br
        raise BridgeException("Bridge does not exist.")
def makeaddress(self, label):
        """Turn a label into an Address with current context.

        Adds repo and path if given a label that only has a :target part.
        """
        addr = address.new(label)
        if not addr.repo:
            addr.repo = self.address.repo
            if not addr.path:
                addr.path = self.address.path
        return addr
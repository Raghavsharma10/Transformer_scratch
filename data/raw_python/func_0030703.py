def temporarySibling(self):
        """
        Create a path naming a temporary sibling of this path in a secure fashion.
        """
        sib = self.parent().child(_secureEnoughString() + self.basename())
        sib.requireCreate()
        return sib
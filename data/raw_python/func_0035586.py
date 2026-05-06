def chival(self, bonds):
        """compute the chiral value around an atom given a list of bonds"""
        # XXX I'm not sure how this works?
        order = [bond.xatom(self) for bond in bonds]
        return self._chirality(order)
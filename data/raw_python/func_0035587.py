def setchival(self, bondorder, rotation):
        """compute chiral ordering of surrounding atoms"""
        rotation = [None, "@", "@@"][(rotation % 2)]
        # check to see if the bonds are attached
        if not bondorder: # use the default xatoms
            if len(self.oatoms) < 3 and self.explicit_hcount != 1:
                raise PinkyError("Need to have an explicit hydrogen when specifying "\
                                  "chirality with less than three bonds")
                

            self._chirality = chirality.T(self.oatoms,
                                            rotation)
            return
        if len(bondorder) != len(self.bonds):
            raise AtomError("The order of all bonds must be specified")
        
        for bond in bondorder:
            if bond not in self.bonds:
                raise AtomError("Specified bonds to assign chirality are not attatched to atom")

        order = [bond.xatom(self) for bond in bonds]
        self._chirality = chirality.T(order, rotation)
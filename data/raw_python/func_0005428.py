def getdilatedmask(self, size=3):
        """
        Returns a morphologically dilated copy of the current mask.
        size = 3 or 5 decides how to dilate.
        """
        if size == 3:
            dilmask = ndimage.morphology.binary_dilation(self.mask, structure=growkernel, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False)
        elif size == 5:
            dilmask = ndimage.morphology.binary_dilation(self.mask, structure=dilstruct, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False)
        else:
            dismask = self.mask.copy()
        
        return dilmask
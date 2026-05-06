def set_stencil_mask(self, mask=8, face='front_and_back'):
        """Control the front or back writing of individual bits in the stencil
    
        Parameters
        ----------
        mask : int
            Mask that is ANDed with ref and stored stencil value.
        face : str
            Can be 'front', 'back', or 'front_and_back'.
        """
        self.glir.command('FUNC', 'glStencilMaskSeparate', face, int(mask))
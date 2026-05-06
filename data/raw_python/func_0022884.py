def set_stencil_func(self, func='always', ref=0, mask=8, 
                         face='front_and_back'):
        """Set front or back function and reference value
    
        Parameters
        ----------
        func : str
            See set_stencil_func.
        ref : int
            Reference value for the stencil test.
        mask : int
            Mask that is ANDed with ref and stored stencil value.
        face : str
            Can be 'front', 'back', or 'front_and_back'.
        """
        self.glir.command('FUNC', 'glStencilFuncSeparate', 
                          face, func, int(ref), int(mask))
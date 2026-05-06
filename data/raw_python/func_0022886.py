def set_stencil_op(self, sfail='keep', dpfail='keep', dppass='keep',
                       face='front_and_back'):
        """Set front or back stencil test actions
    
        Parameters
        ----------
        sfail : str
            Action to take when the stencil fails. Must be one of
            'keep', 'zero', 'replace', 'incr', 'incr_wrap',
            'decr', 'decr_wrap', or 'invert'.
        dpfail : str
            Action to take when the stencil passes.
        dppass : str
            Action to take when both the stencil and depth tests pass,
            or when the stencil test passes and either there is no depth
            buffer or depth testing is not enabled.
        face : str
            Can be 'front', 'back', or 'front_and_back'.
        """
        self.glir.command('FUNC', 'glStencilOpSeparate', 
                          face, sfail, dpfail, dppass)
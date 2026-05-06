def read(self, mode='color', alpha=True):
        """ Return array of pixel values in an attached buffer
        
        Parameters
        ----------
        mode : str
            The buffer type to read. May be 'color', 'depth', or 'stencil'.
        alpha : bool
            If True, returns RGBA array. Otherwise, returns RGB.
        
        Returns
        -------
        buffer : array
            3D array of pixels in np.uint8 format. 
            The array shape is (h, w, 3) or (h, w, 4), with the top-left 
            corner of the framebuffer at index [0, 0] in the returned array.
        
        """
        _check_valid('mode', mode, ['color', 'depth', 'stencil'])
        buffer = getattr(self, mode+'_buffer')
        h, w = buffer.shape[:2]
        
        # todo: this is ostensibly required, but not available in gloo.gl
        #gl.glReadBuffer(buffer._target)
        
        return read_pixels((0, 0, w, h), alpha=alpha)
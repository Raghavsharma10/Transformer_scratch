def resize(self, shape):
        """ Resize all attached buffers with the given shape

        Parameters
        ----------
        shape : tuple of two integers
            New buffer shape (h, w), to be applied to all currently
            attached buffers. For buffers that are a texture, the number
            of color channels is preserved.
        """
        # Check
        if not (isinstance(shape, tuple) and len(shape) == 2):
            raise ValueError('RenderBuffer shape must be a 2-element tuple')
        # Resize our buffers
        for buf in (self.color_buffer, self.depth_buffer, self.stencil_buffer):
            if buf is None:
                continue
            shape_ = shape
            if isinstance(buf, Texture2D):
                shape_ = shape + (self.color_buffer.shape[-1], )
            buf.resize(shape_, buf.format)
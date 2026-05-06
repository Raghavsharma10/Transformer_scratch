def resize(self, shape, format=None):
        """ Set the render-buffer size and format

        Parameters
        ----------
        shape : tuple of integers
            New shape in yx order. A render buffer is always 2D. For
            symmetry with the texture class, a 3-element tuple can also
            be given, in which case the last dimension is ignored.
        format : {None, 'color', 'depth', 'stencil'}
            The buffer format. If None, the current format is maintained. 
            If that is also None, the format will be set upon attaching
            it to a framebuffer. One can also specify the explicit enum:
            GL_RGB565, GL_RGBA4, GL_RGB5_A1, GL_DEPTH_COMPONENT16, or
            GL_STENCIL_INDEX8
        """
        
        if not self._resizeable:
            raise RuntimeError("RenderBuffer is not resizeable")
        # Check shape
        if not (isinstance(shape, tuple) and len(shape) in (2, 3)):
            raise ValueError('RenderBuffer shape must be a 2/3 element tuple')
        # Check format
        if format is None:
            format = self._format  # Use current format (may be None)
        elif isinstance(format, int):
            pass  # Do not check, maybe user needs desktop GL formats
        elif isinstance(format, string_types):
            if format not in ('color', 'depth', 'stencil'):
                raise ValueError('RenderBuffer format must be "color", "depth"'
                                 ' or "stencil", not %r' % format)
        else:
            raise ValueError('Invalid RenderBuffer format: %r' % format)
        
        # Store and send GLIR command
        self._shape = tuple(shape[:2])
        self._format = format
        if self._format is not None:
            self._glir.command('SIZE', self._id, self._shape, self._format)
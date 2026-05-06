def draw(self, mode='triangles', indices=None, check_error=True):
        """ Draw the attribute arrays in the specified mode.

        Parameters
        ----------
        mode : str | GL_ENUM
            'points', 'lines', 'line_strip', 'line_loop', 'triangles',
            'triangle_strip', or 'triangle_fan'.
        indices : array
            Array of indices to draw.
        check_error:
            Check error after draw.
        
        """
        
        # Invalidate buffer (data has already been sent)
        self._buffer = None
        
        # Check if mode is valid
        mode = check_enum(mode)
        if mode not in ['points', 'lines', 'line_strip', 'line_loop',
                        'triangles', 'triangle_strip', 'triangle_fan']:
            raise ValueError('Invalid draw mode: %r' % mode)
        
        # Check leftover variables, warn, discard them
        # In GLIR we check whether all attributes are indeed set
        for name in self._pending_variables:
            logger.warn('Variable %r is given but not known.' % name)
        self._pending_variables = {}
        
        # Check attribute sizes
        attributes = [vbo for vbo in self._user_variables.values() 
                      if isinstance(vbo, DataBuffer)]
        sizes = [a.size for a in attributes]
        if len(attributes) < 1:
            raise RuntimeError('Must have at least one attribute')
        if not all(s == sizes[0] for s in sizes[1:]):
            msg = '\n'.join(['%s: %s' % (str(a), a.size) for a in attributes])
            raise RuntimeError('All attributes must have the same size, got:\n'
                               '%s' % msg)
        
        # Get the glir queue that we need now
        canvas = get_current_canvas()
        assert canvas is not None
        
        # Associate canvas
        canvas.context.glir.associate(self.glir)
        
        # Indexbuffer
        if isinstance(indices, IndexBuffer):
            canvas.context.glir.associate(indices.glir)
            logger.debug("Program drawing %r with index buffer" % mode)
            gltypes = {np.dtype(np.uint8): 'UNSIGNED_BYTE',
                       np.dtype(np.uint16): 'UNSIGNED_SHORT',
                       np.dtype(np.uint32): 'UNSIGNED_INT'}
            selection = indices.id, gltypes[indices.dtype], indices.size
            canvas.context.glir.command('DRAW', self._id, mode, selection)
        elif indices is None:
            selection = 0, attributes[0].size
            logger.debug("Program drawing %r with %r" % (mode, selection))
            canvas.context.glir.command('DRAW', self._id, mode, selection)
        else:
            raise TypeError("Invalid index: %r (must be IndexBuffer)" %
                            indices)
        
        # Process GLIR commands
        canvas.context.flush_commands()
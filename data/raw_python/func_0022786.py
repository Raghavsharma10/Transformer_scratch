def set_texture(self, name, value):
        """ Set a texture sampler. Value is the id of the texture to link.
        """
        if not self._linked:
            raise RuntimeError('Cannot set uniform when program has no code')
        # Get handle for the uniform, first try cache
        handle = self._handles.get(name, -1)
        if handle < 0:
            if name in self._known_invalid:
                return
            handle = gl.glGetUniformLocation(self._handle, name)
            self._unset_variables.discard(name)  # Mark as set
            self._handles[name] = handle  # Store in cache
            if handle < 0:
                self._known_invalid.add(name)
                logger.info('Variable %s is not an active uniform' % name)
                return
        # Program needs to be active in order to set uniforms
        self.activate()
        if True:
            # Sampler: the value is the id of the texture
            tex = self._parser.get_object(value)
            if tex == JUST_DELETED:
                return
            if tex is None:
                raise RuntimeError('Could not find texture with id %i' % value)
            unit = len(self._samplers)
            if name in self._samplers:
                unit = self._samplers[name][-1]  # Use existing unit
            self._samplers[name] = tex._target, tex.handle, unit
            gl.glUniform1i(handle, unit)
def set_uniform(self, name, type_, value):
        """ Set a uniform value. Value is assumed to have been checked.
        """
        if not self._linked:
            raise RuntimeError('Cannot set uniform when program has no code')
        # Get handle for the uniform, first try cache
        handle = self._handles.get(name, -1)
        count = 1
        if handle < 0:
            if name in self._known_invalid:
                return
            handle = gl.glGetUniformLocation(self._handle, name)
            self._unset_variables.discard(name)  # Mark as set
            # if we set a uniform_array, mark all as set
            if not type_.startswith('mat'):
                count = value.nbytes // (4 * self.ATYPEINFO[type_][0])
            if count > 1:
                for ii in range(count):
                    if '%s[%s]' % (name, ii) in self._unset_variables:
                        self._unset_variables.discard('%s[%s]' % (name, ii))

            self._handles[name] = handle  # Store in cache
            if handle < 0:
                self._known_invalid.add(name)
                logger.info('Variable %s is not an active uniform' % name)
                return
        # Look up function to call
        funcname = self.UTYPEMAP[type_]
        func = getattr(gl, funcname)
        # Program needs to be active in order to set uniforms
        self.activate()
        # Triage depending on type
        if type_.startswith('mat'):
            # Value is matrix, these gl funcs have alternative signature
            transpose = False  # OpenGL ES 2.0 does not support transpose
            func(handle, 1, transpose, value)
        else:
            # Regular uniform
            func(handle, count, value)
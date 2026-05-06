def set_attribute(self, name, type_, value):
        """ Set an attribute value. Value is assumed to have been checked.
        """
        if not self._linked:
            raise RuntimeError('Cannot set attribute when program has no code')
        # Get handle for the attribute, first try cache
        handle = self._handles.get(name, -1)
        if handle < 0:
            if name in self._known_invalid:
                return
            handle = gl.glGetAttribLocation(self._handle, name)
            self._unset_variables.discard(name)  # Mark as set
            self._handles[name] = handle  # Store in cache
            if handle < 0:
                self._known_invalid.add(name)
                if value[0] != 0 and value[2] > 0:  # VBO with offset
                    return  # Probably an unused element in a structured VBO
                logger.info('Variable %s is not an active attribute' % name)
                return
        # Program needs to be active in order to set uniforms
        self.activate()
        # Triage depending on VBO or tuple data
        if value[0] == 0:
            # Look up function call
            funcname = self.ATYPEMAP[type_]
            func = getattr(gl, funcname)
            # Set data
            self._attributes[name] = 0, handle, func, value[1:]
        else:
            # Get meta data
            vbo_id, stride, offset = value
            size, gtype, dtype = self.ATYPEINFO[type_]
            # Get associated VBO
            vbo = self._parser.get_object(vbo_id)
            if vbo == JUST_DELETED:
                return
            if vbo is None:
                raise RuntimeError('Could not find VBO with id %i' % vbo_id)
            # Set data
            func = gl.glVertexAttribPointer
            args = size, gtype, gl.GL_FALSE, stride, offset
            self._attributes[name] = vbo.handle, handle, func, args
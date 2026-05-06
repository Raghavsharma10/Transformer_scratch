def set_shaders(self, vert, frag):
        """ This function takes care of setting the shading code and
        compiling+linking it into a working program object that is ready
        to use.
        """
        self._linked = False
        # Create temporary shader objects
        vert_handle = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        frag_handle = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        # For both vertex and fragment shader: set source, compile, check
        for code, handle, type_ in [(vert, vert_handle, 'vertex'),
                                    (frag, frag_handle, 'fragment')]:
            gl.glShaderSource(handle, code)
            gl.glCompileShader(handle)
            status = gl.glGetShaderParameter(handle, gl.GL_COMPILE_STATUS)
            if not status:
                errors = gl.glGetShaderInfoLog(handle)
                errormsg = self._get_error(code, errors, 4)
                raise RuntimeError("Shader compilation error in %s:\n%s" %
                                   (type_ + ' shader', errormsg))
        # Attach shaders
        gl.glAttachShader(self._handle, vert_handle)
        gl.glAttachShader(self._handle, frag_handle)
        # Link the program and check
        gl.glLinkProgram(self._handle)
        if not gl.glGetProgramParameter(self._handle, gl.GL_LINK_STATUS):
            raise RuntimeError('Program linking error:\n%s'
                               % gl.glGetProgramInfoLog(self._handle))
        # Now we can remove the shaders. We no longer need them and it
        # frees up precious GPU memory:
        # http://gamedev.stackexchange.com/questions/47910
        gl.glDetachShader(self._handle, vert_handle)
        gl.glDetachShader(self._handle, frag_handle)
        gl.glDeleteShader(vert_handle)
        gl.glDeleteShader(frag_handle)
        # Now we know what variables will be used by the program
        self._unset_variables = self._get_active_attributes_and_uniforms()
        self._handles = {}
        self._known_invalid = set()
        self._linked = True
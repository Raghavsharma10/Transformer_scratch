def _gl_initialize(self):
        """ Deal with compatibility; desktop does not have sprites
        enabled by default. ES has.
        """
        if '.es' in gl.current_backend.__name__:
            pass  # ES2: no action required
        else:
            # Desktop, enable sprites
            GL_VERTEX_PROGRAM_POINT_SIZE = 34370
            GL_POINT_SPRITE = 34913
            gl.glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
            gl.glEnable(GL_POINT_SPRITE)
        if self.capabilities['max_texture_size'] is None:  # only do once
            self.capabilities['gl_version'] = gl.glGetParameter(gl.GL_VERSION)
            self.capabilities['max_texture_size'] = \
                gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE)
            this_version = self.capabilities['gl_version'].split(' ')[0]
            this_version = LooseVersion(this_version)
def _compute_texture_shape(self, size=1):
        """ Compute uniform texture shape """

        # We should use this line but we may not have a GL context yet
        # linesize = gl.glGetInteger(gl.GL_MAX_TEXTURE_SIZE)
        linesize = 1024
        count = self._uniforms_float_count
        cols = 4 * linesize // int(count)
        rows = max(1, int(math.ceil(size / float(cols))))
        shape = rows, cols * (count // 4), count
        self._ushape = shape
        return shape
def render_to_texture(self, data, texture, offset, size):
        """Render a SDF to a texture at a given offset and size

        Parameters
        ----------
        data : array
            Must be 2D with type np.ubyte.
        texture : instance of Texture2D
            The texture to render to.
        offset : tuple of int
            Offset (x, y) to render to inside the texture.
        size : tuple of int
            Size (w, h) to render inside the texture.
        """
        assert isinstance(texture, Texture2D)
        set_state(blend=False, depth_test=False)

        # calculate the negative half (within object)
        orig_tex = Texture2D(255 - data, format='luminance', 
                             wrapping='clamp_to_edge', interpolation='nearest')
        edf_neg_tex = self._render_edf(orig_tex)

        # calculate positive half (outside object)
        orig_tex[:, :, 0] = data
        
        edf_pos_tex = self._render_edf(orig_tex)

        # render final product to output texture
        self.program_insert['u_texture'] = orig_tex
        self.program_insert['u_pos_texture'] = edf_pos_tex
        self.program_insert['u_neg_texture'] = edf_neg_tex
        self.fbo_to[-1].color_buffer = texture
        with self.fbo_to[-1]:
            set_viewport(tuple(offset) + tuple(size))
            self.program_insert.draw('triangle_strip')
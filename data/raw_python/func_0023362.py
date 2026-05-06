def resize(self, shape, format=None, internalformat=None):
        """Set the texture size and format

        Parameters
        ----------
        shape : tuple of integers
            New texture shape in zyx order. Optionally, an extra dimention
            may be specified to indicate the number of color channels.
        format : str | enum | None
            The format of the texture: 'luminance', 'alpha',
            'luminance_alpha', 'rgb', or 'rgba'. If not given the format
            is chosen automatically based on the number of channels.
            When the data has one channel, 'luminance' is assumed.
        internalformat : str | enum | None
            The internal (storage) format of the texture: 'luminance',
            'alpha', 'r8', 'r16', 'r16f', 'r32f'; 'luminance_alpha',
            'rg8', 'rg16', 'rg16f', 'rg32f'; 'rgb', 'rgb8', 'rgb16',
            'rgb16f', 'rgb32f'; 'rgba', 'rgba8', 'rgba16', 'rgba16f',
            'rgba32f'.  If None, the internalformat is chosen
            automatically based on the number of channels.  This is a
            hint which may be ignored by the OpenGL implementation.
        """
        self._set_emulated_shape(shape)
        Texture2D.resize(self, self._normalize_emulated_shape(shape),
                         format, internalformat)
        self._update_variables()
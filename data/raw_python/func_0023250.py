def _build_interpolation(self):
        """Rebuild the _data_lookup_fn using different interpolations within
        the shader
        """
        interpolation = self._interpolation
        self._data_lookup_fn = self._interpolation_fun[interpolation]
        self.shared_program.frag['get_data'] = self._data_lookup_fn

        # only 'bilinear' uses 'linear' texture interpolation
        if interpolation == 'bilinear':
            texture_interpolation = 'linear'
        else:
            # 'nearest' (and also 'bilinear') doesn't use spatial_filters.frag
            # so u_kernel and shape setting is skipped
            texture_interpolation = 'nearest'
            if interpolation != 'nearest':
                self.shared_program['u_kernel'] = self._kerneltex
                self._data_lookup_fn['shape'] = self._data.shape[:2][::-1]

        if self._texture.interpolation != texture_interpolation:
            self._texture.interpolation = texture_interpolation

        self._data_lookup_fn['texture'] = self._texture

        self._need_interpolation_update = False
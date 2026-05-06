def zoom(self, zoom, center=(0, 0, 0), mapped=True):
        """Update the transform such that its scale factor is changed, but
        the specified center point is left unchanged.

        Parameters
        ----------
        zoom : array-like
            Values to multiply the transform's current scale
            factors.
        center : array-like
            The center point around which the scaling will take place.
        mapped : bool
            Whether *center* is expressed in mapped coordinates (True) or
            unmapped coordinates (False).
        """
        zoom = as_vec4(zoom, default=(1, 1, 1, 1))
        center = as_vec4(center, default=(0, 0, 0, 0))
        scale = self.scale * zoom
        if mapped:
            trans = center - (center - self.translate) * zoom
        else:
            trans = self.scale * (1 - zoom) * center + self.translate
        self._set_st(scale=scale, translate=trans)
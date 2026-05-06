def volume(self, vol, clim=None, method='mip', threshold=None,
               cmap='grays'):
        """Show a 3D volume

        Parameters
        ----------
        vol : ndarray
            Volume to render.
        clim : tuple of two floats | None
            The contrast limits. The values in the volume are mapped to
            black and white corresponding to these values. Default maps
            between min and max.
        method : {'mip', 'iso', 'translucent', 'additive'}
            The render style to use. See corresponding docs for details.
            Default 'mip'.
        threshold : float
            The threshold to use for the isosurafce render style. By default
            the mean of the given volume is used.
        cmap : str
            The colormap to use.

        Returns
        -------
        volume : instance of Volume
            The volume visualization.

        See also
        --------
        Volume
        """
        self._configure_3d()
        volume = scene.Volume(vol, clim, method, threshold, cmap=cmap)
        self.view.add(volume)
        self.view.camera.set_range()
        return volume
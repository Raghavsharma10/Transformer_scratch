def image(self, data, cmap='cubehelix', clim='auto', fg_color=None):
        """Show an image

        Parameters
        ----------
        data : ndarray
            Should have shape (N, M), (N, M, 3) or (N, M, 4).
        cmap : str
            Colormap name.
        clim : str | tuple
            Colormap limits. Should be ``'auto'`` or a two-element tuple of
            min and max values.
        fg_color : Color or None
            Sets the plot foreground color if specified.

        Returns
        -------
        image : instance of Image
            The image.

        Notes
        -----
        The colormap is only used if the image pixels are scalars.
        """
        self._configure_2d(fg_color)
        image = scene.Image(data, cmap=cmap, clim=clim)
        self.view.add(image)
        self.view.camera.aspect = 1
        self.view.camera.set_range()

        return image
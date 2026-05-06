def colorbar(self, cmap, position="right",
                 label="", clim=("", ""),
                 border_width=0.0, border_color="black",
                 **kwargs):
        """Show a ColorBar

        Parameters
        ----------
        cmap : str | vispy.color.ColorMap
            Either the name of the ColorMap to be used from the standard
            set of names (refer to `vispy.color.get_colormap`),
            or a custom ColorMap object.
            The ColorMap is used to apply a gradient on the colorbar.
        position : {'left', 'right', 'top', 'bottom'}
            The position of the colorbar with respect to the plot.
            'top' and 'bottom' are placed horizontally, while
            'left' and 'right' are placed vertically
        label : str
            The label that is to be drawn with the colorbar
            that provides information about the colorbar.
        clim : tuple (min, max)
            the minimum and maximum values of the data that
            is given to the colorbar. This is used to draw the scale
            on the side of the colorbar.
        border_width : float (in px)
            The width of the border the colormap should have. This measurement
            is given in pixels
        border_color : str | vispy.color.Color
            The color of the border of the colormap. This can either be a
            str as the color's name or an actual instace of a vipy.color.Color

        Returns
        -------
        colorbar : instance of ColorBarWidget

        See also
        --------
        ColorBarWidget
        """

        self._configure_2d()

        cbar = scene.ColorBarWidget(orientation=position,
                                    label_str=label,
                                    cmap=cmap,
                                    clim=clim,
                                    border_width=border_width,
                                    border_color=border_color,
                                    **kwargs)

        CBAR_LONG_DIM = 50

        if cbar.orientation == "bottom":
            self.grid.remove_widget(self.cbar_bottom)
            self.cbar_bottom = self.grid.add_widget(cbar, row=5, col=4)
            self.cbar_bottom.height_max = \
                self.cbar_bottom.height_max = CBAR_LONG_DIM

        elif cbar.orientation == "top":
            self.grid.remove_widget(self.cbar_top)
            self.cbar_top = self.grid.add_widget(cbar, row=1, col=4)
            self.cbar_top.height_max = self.cbar_top.height_max = CBAR_LONG_DIM

        elif cbar.orientation == "left":
            self.grid.remove_widget(self.cbar_left)
            self.cbar_left = self.grid.add_widget(cbar, row=2, col=1)
            self.cbar_left.width_max = self.cbar_left.width_min = CBAR_LONG_DIM

        else:  # cbar.orientation == "right"
            self.grid.remove_widget(self.cbar_right)
            self.cbar_right = self.grid.add_widget(cbar, row=2, col=5)
            self.cbar_right.width_max = \
                self.cbar_right.width_min = CBAR_LONG_DIM

        return cbar
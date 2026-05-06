def set_range(self, x=None, y=None, z=None, margin=0.05):
        """ Set the range of the view region for the camera

        Parameters
        ----------
        x : tuple | None
            X range.
        y : tuple | None
            Y range.
        z : tuple | None
            Z range.
        margin : float
            Margin to use.

        Notes
        -----
        The view is set to the given range or to the scene boundaries
        if ranges are not specified. The ranges should be 2-element
        tuples specifying the min and max for each dimension.

        For the PanZoomCamera the view is fully defined by the range.
        For e.g. the TurntableCamera the elevation and azimuth are not
        set. One should use reset() for that.
        """
        # Flag to indicate that this is an initializing (not user-invoked)
        init = self._xlim is None

        # Collect given bounds
        bounds = [None, None, None]
        if x is not None:
            bounds[0] = float(x[0]), float(x[1])
        if y is not None:
            bounds[1] = float(y[0]), float(y[1])
        if z is not None:
            bounds[2] = float(z[0]), float(z[1])
        # If there is no viewbox, store given bounds in lim variables, and stop
        if self._viewbox is None:
            self._set_range_args = bounds[0], bounds[1], bounds[2], margin
            return

        # There is a viewbox, we're going to set the range for real
        self._resetting = True

        # Get bounds from viewbox if not given
        if all([(b is None) for b in bounds]):
            bounds = self._viewbox.get_scene_bounds()
        else:
            for i in range(3):
                if bounds[i] is None:
                    bounds[i] = self._viewbox.get_scene_bounds(i)
        
        # Calculate ranges and margins
        ranges = [b[1] - b[0] for b in bounds]
        margins = [(r*margin or 0.1) for r in ranges]
        # Assign limits for this camera
        bounds_margins = [(b[0]-m, b[1]+m) for b, m in zip(bounds, margins)]
        self._xlim, self._ylim, self._zlim = bounds_margins
        # Store center location
        if (not init) or (self._center is None):
            self._center = [(b[0] + r / 2) for b, r in zip(bounds, ranges)]

        # Let specific camera handle it
        self._set_range(init)

        # Finish
        self._resetting = False
        self.view_changed()
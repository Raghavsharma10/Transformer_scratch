def _set_range(self, init):
        """ Reset the view.
        """

        #PerspectiveCamera._set_range(self, init)

        # Stop moving
        self._speed *= 0.0

        # Get window size (and store factor now to sync with resizing)
        w, h = self._viewbox.size
        w, h = float(w), float(h)

        # Get range and translation for x and y
        x1, y1, z1 = self._xlim[0], self._ylim[0], self._zlim[0]
        x2, y2, z2 = self._xlim[1], self._ylim[1], self._zlim[1]
        rx, ry, rz = (x2 - x1), (y2 - y1), (z2 - z1)

        # Correct ranges for window size. Note that the window width
        # influences the x and y data range, while the height influences
        # the z data range.
        if w / h > 1:
            rx /= w / h
            ry /= w / h
        else:
            rz /= h / w

        # Do not convert to screen coordinates. This camera does not need
        # to fit everything on screen, but we need to estimate the scale
        # of the data in the scene.

        # Set scale, depending on data range. Initial speed is such that
        # the scene can be traversed in about three seconds.
        self._scale_factor = max(rx, ry, rz) / 3.0

        # Set initial position to a corner of the scene
        margin = np.mean([rx, ry, rz]) * 0.1
        self._center = x1 - margin, y1 - margin, z1 + margin

        # Determine initial view direction based on flip axis
        yaw = 45 * self._flip_factors[0]
        pitch = -90 - 20 * self._flip_factors[2]
        if self._flip_factors[1] < 0:
            yaw += 90 * np.sign(self._flip_factors[0])

        # Set orientation
        q1 = Quaternion.create_from_axis_angle(pitch*math.pi/180, 1, 0, 0)
        q2 = Quaternion.create_from_axis_angle(0*math.pi/180, 0, 1, 0)
        q3 = Quaternion.create_from_axis_angle(yaw*math.pi/180, 0, 0, 1)
        #
        self._rotation1 = (q1 * q2 * q3).normalize()
        self._rotation2 = Quaternion()

        # Update
        self.view_changed()
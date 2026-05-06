def viewbox_mouse_event(self, event):
        """ViewBox mouse event handler

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        PerspectiveCamera.viewbox_mouse_event(self, event)

        if event.handled or not self.interactive:
            return

        if event.type == 'mouse_wheel':
            if not event.mouse_event.modifiers:
                # Move forward / backward
                self._speed[0] += 0.5 * event.delta[1]
            elif keys.SHIFT in event.mouse_event.modifiers:
                # Speed
                s = 1.1 ** - event.delta[1]
                self.scale_factor /= s  # divide instead of multiply
                print('scale factor: %1.1f units/s' % self.scale_factor)
            return

        if event.type == 'mouse_press':
            event.handled = True

        if event.type == 'mouse_release':
            # Reset
            self._event_value = None
            # Apply rotation
            self._rotation1 = (self._rotation2 * self._rotation1).normalize()
            self._rotation2 = Quaternion()
        elif not self._timer.running:
            # Ensure the timer runs
            self._timer.start()

        if event.type == 'mouse_move':

            if event.press_event is None:
                return
            if not event.buttons:
                return

            # Prepare
            modifiers = event.mouse_event.modifiers
            pos1 = event.mouse_event.press_event.pos
            pos2 = event.mouse_event.pos
            w, h = self._viewbox.size

            if 1 in event.buttons and not modifiers:
                # rotate

                # get normalized delta values
                d_az = -float(pos2[0] - pos1[0]) / w
                d_el = +float(pos2[1] - pos1[1]) / h
                # Apply gain
                d_az *= - 0.5 * math.pi  # * self._speed_rot
                d_el *= + 0.5 * math.pi  # * self._speed_rot
                # Create temporary quaternions
                q_az = Quaternion.create_from_axis_angle(d_az, 0, 1, 0)
                q_el = Quaternion.create_from_axis_angle(d_el, 1, 0, 0)

                # Apply to global quaternion
                self._rotation2 = (q_el.normalize() * q_az).normalize()

            elif 2 in event.buttons and keys.CONTROL in modifiers:
                # zoom --> fov
                if self._event_value is None:
                    self._event_value = self._fov
                p1 = np.array(event.press_event.pos)[:2]
                p2 = np.array(event.pos)[:2]
                p1c = event.map_to_canvas(p1)[:2]
                p2c = event.map_to_canvas(p2)[:2]
                d = p2c - p1c
                fov = self._event_value * math.exp(-0.01*d[1])
                self._fov = min(90.0, max(10, fov))

        # Make transform be updated on the next timer tick.
        # By doing it at timer tick, we avoid shaky behavior
        self._update_from_mouse = True
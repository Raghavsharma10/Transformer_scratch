def _update_rotation(self, event):
        """Update rotation parmeters based on mouse movement"""
        p2 = event.mouse_event.pos
        if self._event_value is None:
            self._event_value = p2
        wh = self._viewbox.size
        self._quaternion = (Quaternion(*_arcball(p2, wh)) *
                            Quaternion(*_arcball(self._event_value, wh)) *
                            self._quaternion)
        self._event_value = p2
        self.view_changed()
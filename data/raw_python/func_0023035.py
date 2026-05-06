def _update_rotation(self, event):
        """Update rotation parmeters based on mouse movement"""
        p1 = event.mouse_event.press_event.pos
        p2 = event.mouse_event.pos
        if self._event_value is None:
            self._event_value = self.azimuth, self.elevation
        self.azimuth = self._event_value[0] - (p2 - p1)[0] * 0.5
        self.elevation = self._event_value[1] + (p2 - p1)[1] * 0.5
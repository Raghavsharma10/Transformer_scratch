def press_event(self):
        """ The mouse press event that initiated a mouse drag, if any.
        """
        if self.mouse_event.press_event is None:
            return None
        ev = self.copy()
        ev.mouse_event = self.mouse_event.press_event
        return ev
def on_mouse_move(self, event):
        """Mouse move handler

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        if event.is_dragging:
            dxy = event.pos - event.last_event.pos
            button = event.press_event.button

            if button == 1:
                dxy = self.canvas_tr.map(dxy)
                o = self.canvas_tr.map([0, 0])
                t = dxy - o
                self.move(t)
            elif button == 2:
                center = self.canvas_tr.map(event.press_event.pos)
                if self._aspect is None:
                    self.zoom(np.exp(dxy * (0.01, -0.01)), center)
                else:
                    s = dxy[1] * -0.01
                    self.zoom(np.exp(np.array([s, s])), center)
                    
            self.shader_map()